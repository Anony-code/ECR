from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import torch

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    # Safe for paths like "artifacts/xxx.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def apply_json_patch(memory: Dict[str, Any], patch_ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Minimal JSON Patch support for ops we expect:
    - add to list tail with path ending in "/-"
    - replace a scalar/list item
    NOTE: If the model outputs invalid paths, this can throw. Prompts should constrain it.
    """
    for op in patch_ops:
        if op.get("op") not in ("add", "replace"):
            continue

        path = op["path"].strip("/").split("/")
        target = memory

        # Traverse to parent container
        for p in path[:-1]:
            if p.isdigit():
                target = target[int(p)]
            else:
                target = target[p]

        last = path[-1]

        if op["op"] == "add":
            if last == "-":
                target.append(op["value"])
            else:
                if last.isdigit():
                    target[int(last)] = op["value"]
                else:
                    target[last] = op["value"]

        elif op["op"] == "replace":
            if last.isdigit():
                target[int(last)] = op["value"]
            else:
                target[last] = op["value"]

    return memory

def trim_memory_budget(memory: Dict[str, Any], max_templates: int, max_lessons: int) -> Dict[str, Any]:
    """
    Keep experience_memory small to avoid turning it into "statistics" and to control prompt length.
    Policy: keep the most recent items only.
    """
    tmpls = memory["experience_memory"].get("reasoning_templates", [])
    less = memory["experience_memory"].get("calibration_lessons", [])

    if len(tmpls) > max_templates:
        memory["experience_memory"]["reasoning_templates"] = tmpls[-max_templates:]
    if len(less) > max_lessons:
        memory["experience_memory"]["calibration_lessons"] = less[-max_lessons:]

    return memory



def _collect_top_level_json_objects(text: str) -> List[str]:
    """
    Collect all top-level JSON objects {...} in `text` by brace matching.
    Return list of object substrings in order.
    """
    objs: List[str] = []
    start = None
    depth = 0
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i+1])
                    start = None
    return objs

def extract_json(text: str) -> Any:
    """
    Robustly parse JSON from model output.

    Strategy:
    1) If output contains ```json fenced block, parse inside it.
    2) Try parse whole string as JSON.
    3) Collect ALL top-level {...} objects and parse the LAST parseable one.
       (Critical when prompt includes JSON, or model echoes prompt.)
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty output; no JSON found.")

    # 1) fenced block
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p2 = p.strip()
            if p2.startswith("json"):
                candidate = p2[4:].strip()
                if candidate:
                    text = candidate
                    break

    # 2) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3) last JSON object
    objs = _collect_top_level_json_objects(text)
    if not objs:
        raise ValueError("No top-level JSON object found in output.")

    # Try from last to first until one parses
    for js in reversed(objs):
        try:
            return json.loads(js)
        except Exception:
            continue

    raise ValueError("Found JSON-like objects but none were parseable.")

def valid_reasoning_card(obj: Any) -> bool:
    """
    Structural (not semantic) validation for reasoning_card JSON.
    Only checks shape/schema, not medical correctness.
    """
    if not isinstance(obj, dict):
        return False

    required_top_keys = {
        "change_summary",
        "evidence_anchors",
        "mechanistic_hypotheses",
        "ehr_pitfalls_case_specific",
    }
    if not required_top_keys.issubset(obj.keys()):
        return False

    if not isinstance(obj["change_summary"], dict):
        return False
    if not isinstance(obj["evidence_anchors"], dict):
        return False

    mh = obj["mechanistic_hypotheses"]
    if not (isinstance(mh, list) and len(mh) == 2):
        return False

    for h in mh:
        if not isinstance(h, dict):
            return False
        if not {"name", "support", "confidence"}.issubset(h.keys()):
            return False
        if not isinstance(h["support"], list):
            return False

    if not isinstance(obj["ehr_pitfalls_case_specific"], list):
        return False

    return True

def collect_top_level_json_objects_safe(text: str) -> List[str]:
    """
    Collect top-level {...} JSON objects using brace matching,
    ignoring braces inside strings.
    """
    objs = []
    in_str = False
    esc = False
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None

    return objs

def safe_parse_reasoning_card(text: str) -> dict:
    """
    Robustly extract ONE reasoning_card JSON from messy model output.

    Strategy:
    1) Strip sentinel (<END_JSON>) if present.
    2) Try direct json.loads.
    3) Collect all top-level JSON objects (brace-balanced).
    4) Return the FIRST object that passes valid_reasoning_card().
    5) If none pass, raise ValueError.
    """
    if not isinstance(text, str):
        raise ValueError("Model output is not a string.")

    text = text.strip()
    if not text:
        raise ValueError("Empty model output.")

    # 1) strip sentinel if used
    if "<END_JSON>" in text:
        text = text.split("<END_JSON>")[0].strip()

    # 2) direct parse fast-path
    try:
        obj = json.loads(text)
        if valid_reasoning_card(obj):
            return obj
    except Exception:
        pass

    # 3) collect all top-level JSON objects
    candidates = collect_top_level_json_objects_safe(text)
    if not candidates:
        raise ValueError("No JSON object found in model output.")

    parsed_any = False
    for js in candidates:
        try:
            obj = json.loads(js)
            parsed_any = True
            if valid_reasoning_card(obj):
                return obj
        except Exception:
            continue

    if parsed_any:
        raise ValueError("JSON found, but none matched reasoning_card schema.")

    raise ValueError("No parseable JSON object found.")


from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_score, f1_score
)
import numpy as np

def ppv_sensitivity(specificity_levels, _y_true, _y_pred_proba):
    add_sensitivity_results = []
    add_ppv_results =  []
    add_fpr, add_tpr, add_thresholds = roc_curve(_y_true, _y_pred_proba)
    _results = {}
    for specificity in specificity_levels:

        _threshold_index = np.where(add_fpr <= (1 - specificity))[0][-1]
        _threshold = add_thresholds[_threshold_index]

        _sensitivity = add_tpr[_threshold_index]
        add_sensitivity_results.append( _sensitivity)

        _y_pred_binary = (_y_pred_proba >= _threshold).astype(int)
        _ppv = precision_score(_y_true, _y_pred_binary)
        add_ppv_results.append(_ppv)

    _results['Sensitivity'] = add_sensitivity_results
    _results['PPV'] = add_ppv_results

    return _results


def get_evaluation_metrics(labels_all, probs_all, preds_all=None ):

    auroc = roc_auc_score(labels_all, probs_all)
    auprc = average_precision_score(labels_all, probs_all)
    if preds_all is not None:
        f1 = f1_score(labels_all, preds_all, average='binary')
    else:
        f1 = 0
    add_results = ppv_sensitivity([0.9, 0.95], labels_all, probs_all)
    sensitivity_90, sensitivity_95 = add_results['Sensitivity']
    ppv_90, ppv_95 = add_results['PPV']

    return auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95
