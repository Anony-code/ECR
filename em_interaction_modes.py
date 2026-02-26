from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, IterableDataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainerCallback, set_seed
from transformers.utils import logging as hf_logging
from trl import RewardTrainer, RewardConfig

from utils import get_evaluation_metrics, load_json, read_jsonl, write_jsonl
from utils_qwen import qwen_generate, qwen_init
import sys
from scorer_prompts import BASELINE_PROMPT_AD, BASELINE_PROMPT_ADRD, BASELINE_PROMPT_PD
import importlib


_COLOR_INFO = "\033[96m"
_COLOR_WARN = "\033[33m"
_COLOR_RESET = "\033[0m"
_print = print
os.environ["WANDB_DISABLED"]="true"

def _set_all_seeds(seed: int) -> None:
    seed = int(seed)
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print(*args, sep=" ", end="\n", **kwargs):
    _print(*args, sep=sep, end=end, **kwargs)


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _has_adapter(path: str) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def _format_baseline_text(ex, reasoning_text=None, age_offset=5):
    base_dx = ex.get("base_codes_diagnosis", []) or []
    base_md = ex.get("base_codes_medication", []) or []
    delta_dx = ex.get("delta_codes_diagnosis", []) or []
    delta_md = ex.get("delta_codes_medication", []) or []

    base_dx = [i.lower().strip() for i in base_dx]
    base_md = [i.lower().strip() for i in base_md]
    delta_dx = [i.lower().strip() for i in delta_dx]
    delta_md = [i.lower().strip() for i in delta_md]

    parts = [
        f"Age: {ex.get('age') - age_offset};"
        f"Sex: {ex.get('sex')}; "
        f"Baseline diagnoses: {'; '.join(base_dx)}",
        f"Baseline medications: {'; '.join(base_md)}",
        f"Follow-up diagnoses: {'; '.join(delta_dx)}",
        f"Follow-up medications: {'; '.join(delta_md)}",
    ]
    if reasoning_text:
        parts.append("Analysis for this patient: " + str(reasoning_text))

    return "\n".join(parts)


def _build_scorer_text(ex, reasoning_text=None):
    return BASELINE_PROMPT + "\n" + _format_baseline_text(ex, reasoning_text=reasoning_text)


def _get_attr_by_path(obj, path):
    parts = path.split(".")
    for part in parts:
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def _resolve_transformer_layers(model):
    candidates = [
        ("model.layers", "layers"),
        ("model.model.layers", "layers"),
        ("transformer.h", "h"),
        ("gpt_neox.layers", "layers"),
        ("backbone.layers", "layers"),
    ]
    for path, pattern in candidates:
        layers = _get_attr_by_path(model, path)
        if layers is not None:
            return layers, pattern
    return None, None


class LossSwitchRewardTrainer(RewardTrainer):
    def __init__(
        self,
        *args,
        loss_type="pairwise_bce",
        pointwise_alpha=0.2,
        pairwise_margin=0.0,
        pos_weight=None,
        margin=0.1,
        margin_on_sigmoid=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pointwise_alpha = float(pointwise_alpha)
        self.pointwise_pos_weight = None if pos_weight is None else float(pos_weight)
        if self.pointwise_pos_weight is None:
            self.pointwise_bce = torch.nn.BCEWithLogitsLoss()
        else:
            self.pointwise_bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pointwise_pos_weight]),
            )
        self.pairwise_margin = float(pairwise_margin)
        self.margin = float(margin)
        self.margin_on_sigmoid = bool(margin_on_sigmoid)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mode = "train" if self.model.training else "eval"
        inputs["use_cache"] = False
        outputs = model(**inputs)

        chosen_scores, rejected_scores = torch.chunk(outputs.logits.squeeze(-1), chunks=2)
        raw_chosen_scores = chosen_scores
        raw_rejected_scores = rejected_scores

        center_loss = torch.zeros((), device=chosen_scores.device, dtype=chosen_scores.dtype)
        if self.args.center_rewards_coefficient is not None:
            center_loss = self.args.center_rewards_coefficient * torch.mean(
                (chosen_scores + rejected_scores) ** 2
            )

        if hasattr(self, "_metrics") and hasattr(self, "accelerator") and mode in self._metrics:
            with torch.no_grad():
                all_rewards = self.accelerator.gather(
                    torch.cat([raw_chosen_scores, raw_rejected_scores], dim=0)
                )
                self._metrics[mode]["min_reward"].append(all_rewards.min().item())
                self._metrics[mode]["mean_reward"].append(all_rewards.mean().item())
                self._metrics[mode]["max_reward"].append(all_rewards.max().item())

                mean_accuracy = (raw_chosen_scores > raw_rejected_scores).float().mean()
                mean_accuracy = self.accelerator.gather_for_metrics(mean_accuracy).mean().item()
                self._metrics[mode]["accuracy"].append(mean_accuracy)

                mean_margin = (raw_chosen_scores - raw_rejected_scores).mean()
                mean_margin = self.accelerator.gather_for_metrics(mean_margin).mean()
                self._metrics[mode]["margin"].append(mean_margin.item())

        if self.loss_type == "margin":
            if self.margin_on_sigmoid:
                chosen_scores = torch.sigmoid(raw_chosen_scores)
                rejected_scores = torch.sigmoid(raw_rejected_scores)
            loss = torch.relu(self.margin - (chosen_scores - rejected_scores)).mean() + center_loss
            if return_outputs:
                return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
            return loss

        diff = chosen_scores - rejected_scores - self.pairwise_margin
        pairwise_loss = -torch.nn.functional.logsigmoid(diff).mean()
        if self.loss_type == "pairwise":
            pairwise_loss += center_loss
            if return_outputs:
                return pairwise_loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
            return pairwise_loss

        pointwise_targets = torch.cat(
            [torch.ones_like(chosen_scores), torch.zeros_like(rejected_scores)], dim=0
        )
        pointwise_logits = torch.cat([chosen_scores, rejected_scores], dim=0)
        pointwise_loss = self.pointwise_bce(pointwise_logits, pointwise_targets)
        if self.loss_type == "bce":
            loss = pointwise_loss + center_loss
            if return_outputs:
                return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
            return loss

        loss = pairwise_loss + self.pointwise_alpha * pointwise_loss + center_loss
        if return_outputs:
            return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
        return loss


class StepTimerCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=1):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self._step_start = None
        self._first_batch_logged = False

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"train begin | max_steps={state.max_steps} | num_train_epochs={args.num_train_epochs}")

    def on_train_batch_begin(self, args, state, control, **kwargs):
        if self._first_batch_logged:
            return
        inputs = kwargs.get("inputs") or {}
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            try:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                device = str(input_ids.device)
            except Exception:
                batch_size = "NA"
                seq_len = "NA"
                device = "NA"
        else:
            batch_size = "NA"
            seq_len = "NA"
            device = "NA"
        self._first_batch_logged = True

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = datetime.now()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        if self._step_start is None:
            elapsed = "NA"
        else:
            elapsed = (datetime.now() - self._step_start).total_seconds()
            elapsed = f"{elapsed:.2f}s"
        return


class EpochState:
    def __init__(self, base_seed=7):
        self.base_seed = int(base_seed)
        self.counter = 0

    def seed_for_epoch(self):
        seed = self.base_seed + 1000003 * self.counter
        self.counter += 1
        return seed


def build_pref_iterable_dataset_epoch_baseline(
    in_jsonl: str,
    neg_per_pos: int = 1,
    base_seed: int = 7,
    prompt: str | None = None,
    include_meta: bool = False,
    reasoning_map: dict | None = None,
):
    if prompt is None:
        prompt = BASELINE_PROMPT
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string (or None to use default).")

    neg_per_pos = int(neg_per_pos)
    if neg_per_pos <= 0:
        raise ValueError("neg_per_pos must be >= 1")

    raw = _read_jsonl_multi(in_jsonl)

    positives, negatives = [], []
    have_reasoning = 0
    for ex in raw:
        label = int(ex["label"])
        pid = ex.get("id")
        reasoning_text = None
        if reasoning_map is not None and pid is not None:
            reasoning_text = reasoning_map.get(str(pid))
            if reasoning_text is not None:
                have_reasoning += 1

        text = _format_baseline_text(ex, reasoning_text=reasoning_text)
        item = {"pid": pid, "text": text}
        if label == 1:
            positives.append(item)
        else:
            negatives.append(item)

    if not positives or not negatives:
        raise ValueError(f"Need both pos and neg. Got pos={len(positives)}, neg={len(negatives)}")

    print(
        f"\t|Train dataset: cases={len(positives)}, controls={len(negatives)}, "
        f"total={len(positives) + len(negatives)}, with_reasoning={have_reasoning}"
    )

    state = EpochState(base_seed=base_seed)
    first_yield = {"done": False}

    def gen():
        returnseed = state.seed_for_epoch()
        rng = np.random.default_rng(returnseed)
        pos_order = rng.permutation(len(positives))
        for pidx in pos_order:
            pos = positives[int(pidx)]
            for _ in range(neg_per_pos):
                neg = negatives[int(rng.integers(0, len(negatives)))]
                out = {
                    "prompt": prompt,
                    "chosen": pos["text"],
                    "rejected": neg["text"],
                }
                if not first_yield["done"]:
                    first_yield["done"] = True
                if include_meta:
                    out["meta"] = {"pid_chosen": pos["pid"], "pid_rejected": neg["pid"]}
                yield out

    ds = IterableDataset.from_generator(gen)
    return ds, state, len(positives)


def _print_label_stats(prefix, rows):
    pos = 0
    neg = 0
    for ex in rows:
        lab = ex.get("label")
        label = int(lab.item()) if hasattr(lab, "item") else int(lab)
        if label == 1:
            pos += 1
        else:
            neg += 1
    total = pos + neg
    pos_pct = (pos / total * 100.0) if total else 0.0
    neg_pct = (neg / total * 100.0) if total else 0.0
    print(f"{prefix} n={total} | pos={pos} ({pos_pct:.1f}%) | neg={neg} ({neg_pct:.1f}%)")


def _get_label(ex):
    lab = ex.get("label")
    return int(lab.item()) if hasattr(lab, "item") else int(lab)


def _has_pos_neg(rows):
    pos_any = False
    neg_any = False
    for ex in rows:
        label = _get_label(ex)
        if label == 1:
            pos_any = True
        else:
            neg_any = True
        if pos_any and neg_any:
            break
    return pos_any, neg_any


def _select_hard_examples(
    train_rows,
    reasoning_map,
    scorer,
    tokenizer,
    device,
    max_length,
    batch_size,
    apply_sigmoid,
    hard_topk,
    hard_ratio,
    random_topk,
    random_ratio,
):
    texts = []
    labels = []
    idx_map = []
    total_rows = len(train_rows)
    missing_reasoning = 0
    for idx, ex in enumerate(train_rows):
        pid = ex.get("id")
        reasoning_text = None
        if pid is not None and reasoning_map is not None:
            reasoning_text = reasoning_map.get(str(pid))
        if reasoning_text is None:
            missing_reasoning += 1
            continue
        texts.append(_build_scorer_text(ex, reasoning_text=reasoning_text))
        lab = ex.get("label")
        label = int(lab.item()) if hasattr(lab, "item") else int(lab)
        labels.append(label)
        idx_map.append(idx)

    if not texts:
        print(
            f"\thard pool empty | rows={total_rows} | missing_reasoning={missing_reasoning}"
        )
        return []
    pos_n = sum(1 for y in labels if y == 1)
    neg_n = sum(1 for y in labels if y == 0)
    print(
        f"\thard pool size={len(texts)} | pos={pos_n} | neg={neg_n}"
    )

    scores = _score_texts(
        scorer,
        tokenizer,
        texts,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        apply_sigmoid=apply_sigmoid,
    )
    pos = [(idx_map[i], scores[i]) for i, y in enumerate(labels) if y == 1]
    neg = [(idx_map[i], scores[i]) for i, y in enumerate(labels) if y == 0]

    def _pick(n, items, reverse):
        if not items or n <= 0:
            return []
        items_sorted = sorted(items, key=lambda x: x[1], reverse=reverse)
        return [i for i, _ in items_sorted[:n]]

    if hard_topk and int(hard_topk) > 0:
        k_pos = min(int(hard_topk), len(pos))
        k_neg = min(int(hard_topk), len(neg))
    else:
        ratio = max(0.0, min(1.0, float(hard_ratio)))
        k_pos = min(max(1, int(len(pos) * ratio)), len(pos)) if pos else 0
        k_neg = min(max(1, int(len(neg) * ratio)), len(neg)) if neg else 0

    hard_pos = _pick(k_pos, pos, reverse=False)  # pos: lower score is harder
    hard_neg = _pick(k_neg, neg, reverse=True)   # neg: higher score is harder
    hard = list(dict.fromkeys(hard_pos + hard_neg))

    if (random_topk and int(random_topk) > 0) or (random_ratio and float(random_ratio) > 0.0):
        rng = np.random.default_rng(17)
        hard_set = set(hard)
        remaining = [i for i in idx_map if i not in hard_set]
        if remaining:
            if random_topk and int(random_topk) > 0:
                k_rand = min(int(random_topk), len(remaining))
            else:
                ratio = max(0.0, min(1.0, float(random_ratio)))
                k_rand = min(max(1, int(len(remaining) * ratio)), len(remaining))
            rand_pick = rng.choice(remaining, size=k_rand, replace=False).tolist()
            hard = list(dict.fromkeys(hard + rand_pick))

    print(
        f"\thard selection | hard_pos={len(hard_pos)} | hard_neg={len(hard_neg)} | "
        f"random_added={(len(hard) - len(hard_pos) - len(hard_neg))}"
    )
    return hard


def _read_jsonl_multi(paths):
    if isinstance(paths, (list, tuple)):
        rows = []
        for p in paths:
            rows.extend(read_jsonl(str(p)))
        return rows
    return read_jsonl(str(paths))


def _find_jsonl_files(data_dir: Path, pattern: str):
    data_dir = Path(data_dir)
    matched = sorted(data_dir.glob(pattern))
    if not matched:
        raise FileNotFoundError(f"No files found with pattern {pattern} in {data_dir}")
    return matched


def _ensure_pos_neg_selected(selected, train_rows, seed):
    pos_idx, neg_idx = [], []
    for i, ex in enumerate(train_rows):
        if _get_label(ex) == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)
    rng = np.random.default_rng(int(seed))
    if not selected:
        out = []
        if pos_idx:
            out.append(int(rng.choice(pos_idx)))
        if neg_idx:
            cand = [i for i in neg_idx if i not in out]
            if cand:
                out.append(int(rng.choice(cand)))
        return out
    selected_set = set(selected)
    has_pos = any((_get_label(train_rows) == 1) for i in selected)
    has_neg = any((_get_label(train_rows) == 0) for i in selected)
    if not has_pos and pos_idx:
        cand = [i for i in pos_idx if i not in selected_set]
        if cand:
            selected.append(int(rng.choice(cand)))
            selected_set.add(selected[-1])
    if not has_neg and neg_idx:
        cand = [i for i in neg_idx if i not in selected_set]
        if cand:
            selected.append(int(rng.choice(cand)))
            selected_set.add(selected[-1])
    return selected


def _score_texts(
    model,
    tokenizer,
    texts,
    device,
    max_length=5000,
    batch_size=8,
    apply_sigmoid=False,
):
    model.eval()
    scores = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            batch_scores = out.logits.squeeze(-1)
            if apply_sigmoid:
                batch_scores = torch.sigmoid(batch_scores)
            batch_scores = batch_scores.detach().float().cpu().view(-1).tolist()
            scores.extend(batch_scores)
    return scores


def build_pointwise_dataset_baseline(in_jsonl, reasoning_map: dict | None = None):
    if isinstance(in_jsonl, list):
        raw = _read_jsonl_multi(in_jsonl)
    else:
        raw = _read_jsonl(in_jsonl)
    items = []
    prompt = BASELINE_PROMPT
    positives, negatives = [], []
    have_reasoning =0
    for ex in raw:
        pid = ex.get('id')
        reasoning_text = None
        if reasoning_map is not None and pid is not None:
            reasoning_text = reasoning_map.get(str(pid))
            if reasoning_text is not None:
                have_reasoning += 1
        text = _format_baseline_text(ex, reasoning_text=reasoning_text)

        text = prompt + "\n" + text
        items.append(
            {
                "text":text,
                "label": int(ex["label"]),
            }
        )

        label = int(ex["label"])
        if label == 1:
            positives.append(pid)
        else:
            negatives.append(pid)

    print(
        f"\t|Test dataset: cases={len(positives)}, controls={len(negatives)}, "
        f"total={len(positives) + len(negatives)}, with_reasoning={have_reasoning}"
    )

    return Dataset.from_list(items)


def _pretokenize_pointwise_dataset(test_ds, tokenizer, max_length, batch_size=256):
    items = []
    n = len(test_ds)
    for i in range(0, n, batch_size):
        batch = test_ds[i : i + batch_size]
        enc = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        for j in range(len(batch["text"])):
            items.append(
                {
                    "input_ids": enc["input_ids"][j],
                    "attention_mask": enc["attention_mask"][j],
                    "label": int(batch["label"][j]),
                }
            )
    ds = Dataset.from_list(items)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds


def eval_pointwise(
    model,
    tokenizer,
    test_dataset,
    text_key="text",
    label_key="label",
    batch_size=8,
    max_length=10000,
    device=None,
    apply_sigmoid=False,
    collator=None,
):
    model.eval()
    scores_all, labels_all = [], []
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    n = len(test_dataset)
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            batch = test_dataset[i : i + batch_size]
            labels = batch[label_key]
            if "input_ids" in batch:
                features = [
                    {
                        "input_ids": batch["input_ids"][j],
                        "attention_mask": batch["attention_mask"][j],
                    }
                    for j in range(len(labels))
                ]
                if collator is None:
                    collator = DataCollatorWithPadding(
                        tokenizer=tokenizer,
                        padding="longest",
                        return_tensors="pt",
                    )
                enc = collator(features)
            else:
                texts = batch[text_key]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=max_length,
                )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            scores = out.logits.squeeze(-1)
            if apply_sigmoid:
                scores = torch.sigmoid(scores)

            scores_all.append(scores.detach().float().cpu())
            if isinstance(labels, torch.Tensor):
                labels_all.append(labels.detach().to(dtype=torch.int64).cpu())
            else:
                labels_all.append(torch.tensor(labels, dtype=torch.int64))

    scores = torch.cat(scores_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return scores, labels


class PeriodicEvalCallback(TrainerCallback):
    def __init__(
        self,
        test_jsonl,
        tokenizer,
        max_length,
        batch_size,
        eval_every_steps,
        apply_sigmoid,
        reasoning_map=None,
    ):
        self.test_jsonl = test_jsonl
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.eval_every_steps = eval_every_steps
        self.apply_sigmoid = apply_sigmoid
        self.test_ds = None
        self.reasoning_map = reasoning_map

        self.eval_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            return_tensors="pt",
        )

    def on_train_begin(self, args, state, control, **kwargs):
        if self.test_jsonl:
            raw_ds = build_pointwise_dataset_baseline(
                self.test_jsonl,
                reasoning_map=self.reasoning_map,
            )
            self.test_ds = _pretokenize_pointwise_dataset(
                raw_ds,
                self.tokenizer,
                self.max_length,
            )
        if not self.test_ds:
            return
        model = kwargs["model"]
        model.eval()
        scores, labels = eval_pointwise(
            model,
            self.tokenizer,
            self.test_ds,
            text_key="text",
            label_key="label",
            batch_size=self.batch_size,
            max_length=self.max_length,
            apply_sigmoid=self.apply_sigmoid,
            collator=self.eval_collator
        )
        auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95 = (
            get_evaluation_metrics(labels, scores)
        )
        print(
            f"[Eval@{state.global_step}] AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"F1={f1:.4f} Sens@90%={sensitivity_90:.4f} Sens@95%={sensitivity_95:.4f} "
            f"PPV@90%={ppv_90:.4f} PPV@95%={ppv_95:.4f}"
        )
        model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if not self.test_ds:
            return control
        if state.global_step > 0 and state.global_step % self.eval_every_steps == 0:
            model = kwargs["model"]
            model.eval()
            scores, labels = eval_pointwise(
                model,
                self.tokenizer,
                self.test_ds,
                text_key="text",
                label_key="label",
                batch_size=self.batch_size,
                max_length=self.max_length,
                apply_sigmoid=self.apply_sigmoid,
                collator=self.eval_collator,
            )
            auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95 = (
                get_evaluation_metrics(labels, scores)
            )
            print(
                f"[Eval@{state.global_step}] AUROC={auroc:.4f} AUPRC={auprc:.4f} "
                f"F1={f1:.4f} Sens@90%={sensitivity_90:.4f} Sens@95%={sensitivity_95:.4f} "
                f"PPV@90%={ppv_90:.4f} PPV@95%={ppv_95:.4f}"
            )
            model.train()
        return control

    def single_eval(self, model, step):
        model.eval()
        scores, labels = eval_pointwise(
            model,
            self.tokenizer,
            self.test_ds,
            text_key="text",
            label_key="label",
            batch_size=self.batch_size,
            max_length=self.max_length,
            apply_sigmoid=self.apply_sigmoid,
            collator=self.eval_collator,
        )
        auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95 = (
            get_evaluation_metrics(labels, scores)
        )
        print(
            f"[Eval@{step}] AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"F1={f1:.4f} Sens@90%={sensitivity_90:.4f} Sens@95%={sensitivity_95:.4f} "
            f"PPV@90%={ppv_90:.4f} PPV@95%={ppv_95:.4f}"
        )
        model.train()


def _init_generator(gen_model_id, device_map, max_memory=None, dtype="auto"):
    print('load generator from phase 1', gen_model_id, '\n')
    if gen_model_id == "Qwen/Qwen3-4B-Instruct-2507":
        gen, tokenizer = qwen_init(
            device_map=device_map,
            max_memory=max_memory,
            dtype=dtype,
        )
        return gen, tokenizer
    if gen_model_id == "openai/gpt-oss-20b":
        gen = pipeline(
            "text-generation",
            model=gen_model_id,
            device_map=device_map,
            torch_dtype=dtype,
        )
        return gen, None
    raise ValueError(f"Unsupported gen_model_id: {gen_model_id}")


def _generate_reasoning(
    gen,
    tokenizer,
    gen_model_id,
    prompt,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
):
    mess = [
        {"role": "system", "content": "Reasoning: low"},
        {"role": "user", "content": prompt},
    ]
    if gen_model_id == "Qwen/Qwen3-4B-Instruct-2507":
        return qwen_generate(
            gen,
            tokenizer,
            mess,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
    rc_text = gen(
        mess,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_full_text=False,
    )[0]["generated_text"].strip()
    return rc_text.split("assistantfinal")[-1]


def _train_scorer_from_subset(
    subset_jsonl,
    reasoning_rows,
    eval_reasoning_map,
    output_dir,
    model_path,
    base_model_name,
    dtype,
    seed,
    lr,
    epochs,
    batch_size,
    grad_accum,
    max_length,
    neg_per_pos,
    save_steps,
    score_only,
    lora_last_n,
    loss_type,
    pointwise_alpha,
    pairwise_margin,
    pos_weight,
    margin,
    margin_on_sigmoid,
    device,
    eval_jsonl,
    eval_steps,
    eval_batch_size,
    eval_sigmoid,
):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    torch_dtype = dtype_map.get(str(dtype), torch.bfloat16)
    model, used_adapter = _load_full_model_for_training(
        model_path,
        base_model_name,
        torch_dtype,
    )
    model.to(device)

    tok_path = _resolve_tokenizer_path(model_path, base_model_name if used_adapter else model_path)
    tok = AutoTokenizer.from_pretrained(
        tok_path,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.config.pad_token_id = tok.pad_token_id

    reasoning_map = {}
    for row in reasoning_rows:
        pid = row.get("id")
        text = row.get("reasoning")
        if pid is not None and text is not None:
            reasoning_map[str(pid)] = str(text)

    subset_rows = read_jsonl(str(subset_jsonl))
    covered = 0
    for ex in subset_rows:
        pid = ex.get("id")
        if pid is not None and str(pid) in reasoning_map:
            covered += 1
    print(f"\n\ttrainer reasoning coverage: {covered}/{len(subset_rows)}")

    dy_ds, _epoch_state, pos_count = build_pref_iterable_dataset_epoch_baseline(
        in_jsonl=str(subset_jsonl),
        neg_per_pos=neg_per_pos,
        base_seed=seed,
        prompt=None,
        include_meta=False,
        reasoning_map=reasoning_map,
    )

    pairs_per_epoch = pos_count * neg_per_pos
    steps_per_epoch = math.ceil(pairs_per_epoch / (batch_size * grad_accum))
    max_steps = steps_per_epoch * epochs

    print(
        f"\ttrain steps | pos={pos_count} | neg_per_pos={neg_per_pos} | "
        f"pairs/epoch={pairs_per_epoch} | bs={batch_size} | grad_accum={grad_accum} | "
        f"steps/epoch={steps_per_epoch} | max_steps={max_steps}"
    )

    rcfg = RewardConfig(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_length=max_length,
        logging_strategy="no",
        disable_tqdm=False,
        save_strategy="steps" if save_steps > 0 else "no",
        save_steps=max(save_steps, 1),
        eval_strategy="no",
        center_rewards_coefficient=1e-2,
        dataloader_num_workers=0,
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )

    peft_config = None
    has_existing_adapter = hasattr(model, "peft_config")
    if score_only and (lora_last_n > 0):
        print("Warning: score_only and last-n LoRA both set; using score_only.")
    if score_only:
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "score" in name:
                if "modules_to_save" in name:
                    p.requires_grad = True
                elif not has_existing_adapter:
                    p.requires_grad = True
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        warnings.filterwarnings(
            "ignore",
            message="None of the inputs have requires_grad=True. Gradients will be None",
            category=UserWarning,
        )
        trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(
                "score_only selected but no trainable score parameters found. "
                "This usually means the adapter does not include modules_to_save for score."
            )
    else:
        if has_existing_adapter:
            for p in model.parameters():
                p.requires_grad = False
            if hasattr(model, "enable_adapter_layers"):
                model.enable_adapter_layers()
            for name, p in model.named_parameters():
                if "lora_" in name or "adapter" in name:
                    p.requires_grad = True
                if "score" in name:
                    if "modules_to_save" in name:
                        p.requires_grad = True
            trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        else:
            if lora_last_n > 0:
                layers, layers_pattern = _resolve_transformer_layers(model)
                if layers is None:
                    raise ValueError("Unable to locate transformer layers for --lora_last_n.")
                num_layers = len(layers)
                n_last = min(lora_last_n, num_layers)
                layers_to_transform = list(range(num_layers - n_last, num_layers))
                lora = LoraConfig(
                    r=lora.r,
                    lora_alpha=lora.lora_alpha,
                    lora_dropout=lora.lora_dropout,
                    bias=lora.bias,
                    target_modules=lora.target_modules,
                    modules_to_save=lora.modules_to_save,
                    layers_pattern=layers_pattern,
                    layers_to_transform=layers_to_transform,
                )
            if lora_last_n > 0:
                print("\n\tNew last-n LoRA training...")
            else:
                print("\n\tNew full-layer LoRA training...")
            peft_config = lora


    trainer = LossSwitchRewardTrainer(
        model=model,
        args=rcfg,
        train_dataset=dy_ds,
        processing_class=tok,
        peft_config=peft_config,
        loss_type=loss_type,
        pointwise_alpha=pointwise_alpha,
        pairwise_margin=pairwise_margin,
        pos_weight=pos_weight,
        margin=margin,
        margin_on_sigmoid=margin_on_sigmoid,
    )
    eval_callback = None
    if eval_jsonl is not None and int(eval_steps) > 0:
        eval_callback = PeriodicEvalCallback(
            test_jsonl=eval_jsonl,
            tokenizer=tok,
            max_length=max_length,
            batch_size=eval_batch_size,
            eval_every_steps=int(eval_steps),
            apply_sigmoid=bool(eval_sigmoid),
            reasoning_map=eval_reasoning_map
        )
        trainer.add_callback(eval_callback)


    if score_only:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tTrainable params: {trainable}/{total} ({trainable / total * 100:.4f}%) [score_only]")
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\tTrainable params: {trainable}/{total} ({trainable / total * 100:.4f}%) [lora_last_n={lora_last_n}]")

    print("\n\tStart training...")
    trainer.train()
    if eval_callback is not None and eval_callback.test_ds:
        eval_callback.single_eval(trainer.model, trainer.state.global_step)

    if score_only:
        print('\tSave full model to', output_dir)
        trainer.model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
    else:
        print('\tSave PEFT model to', output_dir)
        if not hasattr(trainer.model, "save_pretrained"):
            raise ValueError("Expected model with save_pretrained for LoRA saving.")
        trainer.model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)


def _load_scorer_for_scoring(scorer_path, base_model_name, scorer_dtype):

    if scorer_path is None:
        raise ValueError("scorer_path is None")

    scorer_path = os.path.abspath(scorer_path)

    if _has_adapter(scorer_path):
        scorer = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            dtype=scorer_dtype,
        )
        scorer = PeftModel.from_pretrained(scorer, scorer_path)
        return scorer

    if os.path.isdir(scorer_path):
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                scorer_path,
                num_labels=1,
                dtype=scorer_dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"\tLocal load failed ({e}). Fallback to base config.")
            cfg = AutoConfig.from_pretrained(
                base_model_name,
                num_labels=1,
                trust_remote_code=True,
            )
            return AutoModelForSequenceClassification.from_pretrained(
                scorer_path,
                config=cfg,
                dtype=scorer_dtype,
                local_files_only=True,
                trust_remote_code=True,
            )

    return AutoModelForSequenceClassification.from_pretrained(
        scorer_path,
        num_labels=1,
        dtype=scorer_dtype,
        trust_remote_code=True,
    )


def _load_full_model_for_training(model_path, base_model_name, torch_dtype):
    if _has_adapter(model_path):
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            dtype=torch_dtype,
        )
        model = PeftModel.from_pretrained(base, model_path)
        print("\tLoad adapter for continued training")
        return model, True

    print("\n\tLoad full model for training")
    return AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        dtype=torch_dtype,
    ), False


def _resolve_tokenizer_path(preferred_path, fallback_path):
    if preferred_path is not None:
        p = Path(preferred_path)
        if (p / "tokenizer.json").exists() or (p / "tokenizer_config.json").exists():
            return preferred_path

    return fallback_path


def main():
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    ap = argparse.ArgumentParser()


    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--device_map", type=str, default="cuda:0")
    ap.add_argument("--gen_dtype", type=str, default="bfloat16")
    ap.add_argument("--scorer_dtype", type=str, default="bfloat16")
    ap.add_argument("--train_dtype", type=str, default="bfloat16")

    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--reasoning_jsonl", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--reasoning_pid_key", type=str, default="id")
    ap.add_argument("--reasoning_text_key", type=str, default="reasoning")


    ap.add_argument("--memory_path", type=str, required=True)
    ap.add_argument("--gen_model_id", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max_memory", type=str, default=None)
    ap.add_argument("--max_new_tokens_reasoning", type=int, default=512)
    ap.add_argument("--rollout_n", type=int, default=3)
    ap.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)


    ap.add_argument("--scorer_model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--scorer_lora_dir", type=str, required=True)
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--score_max_length", type=int, default=5000)
    ap.add_argument("--score_sigmoid", action=argparse.BooleanOptionalAction, default=True)


    ap.add_argument("--preroll_path", type=str, default="phase3_em_0127_161412_51_r1")

    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["rollout_only", "score_only", "rollout_and_score", "preroll"],
        help="phase3 mode. If None, will infer from --preroll and whether scorer is needed.",
    )

    ap.add_argument("--random_subset", action="store_true")
    ap.add_argument("--subset_pos_ratio", type=float, default=None)
    ap.add_argument("--select_topk", type=int, default=None)
    ap.add_argument("--select_ratio", type=float, default=0.005)
    ap.add_argument("--select_threshold", type=float, default=None)
    ap.add_argument("--em_rounds", type=int, default=1)


    ap.add_argument("--out_dir", type=str, default="artifacts/phase3")
    ap.add_argument("--runnote", type=str, default="n")
    ap.add_argument("--seed", type=int, default=7)


    ap.add_argument("--train_scorer", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train_output_dir", type=str, default="artifacts/phase3/scorer_ckpt")
    ap.add_argument("--train_lr", type=float, default=1e-3)
    ap.add_argument("--train_epochs", type=int, default=2)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--train_grad_accum", type=int, default=32)
    ap.add_argument("--train_max_length", type=int, default=5000)
    ap.add_argument("--train_neg_per_pos", type=int, default=2)
    ap.add_argument("--train_save_steps", type=int, default=0)
    ap.add_argument("--train_scorer_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--train_lora_last_n", type=int, default=0)
    ap.add_argument("--eval_jsonl", type=str, default=None)
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_sigmoid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dataset", type=str, default="ad", choices=["ad", "pd", "adrd"])
    ap.add_argument(
        "--train_loss_type",
        type=str,
        default="pairwise_bce",
        choices=["pairwise", "pairwise_bce", "margin", "bce"],
    )
    ap.add_argument("--train_pointwise_alpha", type=float, default=0.2)

    ap.add_argument("--train_pairwise_margin", type=float, default=0.0)
    ap.add_argument("--train_pos_weight", type=float, default=None)

    ap.add_argument("--train_margin", type=float, default=0.1)
    ap.add_argument(
        "--train_margin_on_sigmoid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


    args = ap.parse_args()
    if args.mode is None:
        print('No mode. Exit.')
        sys.exit()
    print(f"[Info] mode={args.mode}, dataset={args.dataset}")


    _set_all_seeds(args.seed)


    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    rand = random.randint(10, 99)
    run_id = f"{args.dataset}_{args.runnote}_{args.mode}_{timestamp}_{rand}"


    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": torch.bfloat16,}
    scorer_dtype = dtype_map.get(args.scorer_dtype, torch.bfloat16)


    global BASELINE_PROMPT
    if args.dataset =='ad':
        BASELINE_PROMPT = BASELINE_PROMPT_AD
    if args.dataset =='adrd':
        BASELINE_PROMPT = BASELINE_PROMPT_ADRD
    if args.dataset =='pd':
        BASELINE_PROMPT = BASELINE_PROMPT_PD
        print('\nTake BASELINE_PROMPT for PD')

    try:
        module = importlib.import_module(f"prompts.prompts_{args.dataset}")
        clinical_reasoning_prompt = module.clinical_reasoning_prompt

        from memory_schema import init_memory, init_memory_adrd, init_memory_pd
        INIT_MEMORY = {
            "adrd": init_memory_adrd,
            "pd": init_memory_pd,
            "ad": init_memory,
        }
        init_memory = INIT_MEMORY.get(args.dataset)
    except:
        raise ValueError(f"Unsupported cohort prompt: {args.dataset}")


    train = read_jsonl(args.train_jsonl)
    print("training data loaded:", len(train))
    _print_label_stats("train label stats:", train)
    train_by_id = {str(ex.get("id")): ex for ex in train if ex.get("id") is not None}

    eval_jsonl = args.eval_jsonl

    reasoning_map = {}
    if args.reasoning_jsonl:
        if args.dataset == "ad":
            ratio = 10
            train_jsonl_reasoning = f"artifacts/phase1/{args.dataset}/train_set_{ratio}_reasoning.jsonl"
            train_reasoning_rows = _read_jsonl(train_jsonl_reasoning)
            test_jsonl_reasoning = _find_jsonl_files(
                data_dir=f"artifacts/phase1/{args.dataset}",
                pattern="test_set_reasoning",
            )
            test_reasoning_rows = _read_jsonl_multi(test_jsonl_reasoning)

        elif args.dataset == "pd":
            ratio = 5
            train_jsonl_reasoning = f"artifacts/phase1/{args.dataset}/train_set_{ratio}_reasoning.jsonl"
            train_reasoning_rows = _read_jsonl(train_jsonl_reasoning)
            test_jsonl_reasoning = _find_jsonl_files(
                data_dir=f"artifacts/phase1/{args.dataset}",
                pattern="test_set_reasoning",
            )
            test_reasoning_rows = _read_jsonl_multi(test_jsonl_reasoning)

        elif args.dataset == "adrd":
            ratio = 10
            train_jsonl_reasoning = f"artifacts/phase1/{args.dataset}/train_set_{ratio}_reasoning.json"
            train_reasoning_rows = _read_jsonl(train_jsonl_reasoning)
            test_jsonl_reasoning = _find_jsonl_files(
                data_dir=f"artifacts/phase1/{args.dataset}",
                pattern="test_set_reasoning",
            )
            test_reasoning_rows = _read_jsonl_multi(test_jsonl_reasoning)

        for row in train_reasoning_rows:
            pid = row.get(args.reasoning_pid_key)
            text = row.get(args.reasoning_text_key)
            if pid is not None and text is not None:
                reasoning_map[str(pid)] = str(text)
        rsum = len(reasoning_map)
        print(f"[Info] reasoning_map training entries: {rsum}")

        for row in test_reasoning_rows:
            pid = row.get(args.reasoning_pid_key)
            text = row.get(args.reasoning_text_key)
            if pid is not None and text is not None:
                reasoning_map[str(pid)] = str(text)
        print(f"[Info] reasoning_map testing entries: {len(reasoning_map) - rsum}")
        print(f"[Info] reasoning_map total entries: {len(reasoning_map)}")
    else:
        print("[Info] no reasoning files loaded; reasoning_map={}")

    memory = None
    try:
        memory = load_json(args.memory_path)
        print("\nload memory from:", args.memory_path)
    except Exception:
        memory = init_memory()
        print("\nmemory init_memory (fallback)")

    max_memory = None
    if args.max_memory:
        try:
            max_memory = json.loads(args.max_memory)
        except json.JSONDecodeError:
            raise ValueError("--max_memory must be a valid JSON string")


    print("\nLoad generator...")
    gen, gen_tok = _init_generator(
        args.gen_model_id,
        args.device_map,
        max_memory=max_memory,
        dtype=args.gen_dtype,
    )


    current_scorer_path = args.scorer_lora_dir
    base_model_name = args.scorer_model_name
    train_model_path = args.scorer_lora_dir

    evalfirst=False
    if evalfirst:
        print("\nInitial eval before EM...")
        tok_path = _resolve_tokenizer_path(args.scorer_lora_dir, args.scorer_model_name)
        tok = AutoTokenizer.from_pretrained(
            tok_path,
            use_fast=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"


        scorer_eval = _load_scorer_for_scoring(
            current_scorer_path,
            base_model_name,
            scorer_dtype,
        )
        scorer_eval.to(device)
        scorer_eval.eval()
        scorer_eval.config.pad_token_id = tok.pad_token_id
        eval_ds = build_pointwise_dataset_baseline(eval_jsonl, reasoning_map=reasoning_map)


        scores, labels = eval_pointwise(
            scorer_eval,
            tok,
            eval_ds,
            text_key="text",
            label_key="label",
            batch_size=args.eval_batch_size,
            max_length=args.score_max_length,
            apply_sigmoid=bool(args.eval_sigmoid),
        )
        auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95 = (
            get_evaluation_metrics(labels, scores)
        )
        print(
            f"\tinitial eval | AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"F1={f1:.4f} Sens@90%={sensitivity_90:.4f} Sens@95%={sensitivity_95:.4f} "
            f"PPV@90%={ppv_90:.4f} PPV@95%={ppv_95:.4f}"
        )


    for r in range(int(args.em_rounds)):
        print(f"\nRound {r+1}")
        round_tag = f"r{r+1}"
        out_dir = Path(args.out_dir) / f"{run_id}_{round_tag}"
        os.makedirs(out_dir, exist_ok=True)
        print(f"\tOutput dir: {out_dir}")


        if args.mode in ["rollout_only", "rollout_and_score"]:
            need_scorer = (args.mode == "rollout_and_score")

            tok = None
            scorer = None
            if need_scorer:
                tok_path = _resolve_tokenizer_path(args.scorer_lora_dir, args.scorer_model_name)
                tok = AutoTokenizer.from_pretrained(
                    tok_path,
                    use_fast=True,
                )
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "right"

                scorer = _load_scorer_for_scoring(
                    current_scorer_path,
                    base_model_name,
                    scorer_dtype,
                )
                scorer.to(device)
                scorer.eval()
                scorer.config.pad_token_id = tok.pad_token_id

            print("\n\tSelect training samples for rollout...")

            if args.mode == "rollout_only":
                if not args.random_subset:
                    print("rollout_only forces random_subset=True")
                    args.random_subset = True

            indexed_scores = None

            if args.random_subset:
                if args.select_threshold is not None:
                    print("\trandom_subset ignores --select_threshold")
                if args.select_topk is not None:
                    k = max(1, int(args.select_topk))
                elif args.select_ratio is not None:
                    k = max(1, int(len(train) * float(args.select_ratio)))
                else:
                    k = max(1, int(len(train) * 0.2))
                k = min(k, len(train))

                rng = np.random.default_rng(int(args.seed) + int(r))
                pos_idx, neg_idx = [], []
                for i, ex in enumerate(train):
                    label = _get_label(ex)
                    if label == 1:
                        pos_idx.append(i)
                    else:
                        neg_idx.append(i)

                if pos_idx and neg_idx:
                    if k < 2:
                        k = 2
                    if args.subset_pos_ratio is not None:
                        print(f"\tuse subset_pos_ratio {args.subset_pos_ratio} when doing random subset sampling")
                        pos_ratio = float(args.subset_pos_ratio)
                        k_pos = int(round(k * pos_ratio))
                        k_neg = k - k_pos
                        if k_pos <= 0 or k_neg <= 0:
                            raise ValueError("subset_pos_ratio must yield both pos and neg when available")
                        if k_pos > len(pos_idx) or k_neg > len(neg_idx):
                            raise ValueError("subset_pos_ratio exceeds available pos/neg samples")
                        pos_pick = rng.choice(pos_idx, size=k_pos, replace=False).tolist()
                        neg_pick = rng.choice(neg_idx, size=k_neg, replace=False).tolist()
                        selected = pos_pick + neg_pick
                    else:
                        pos_choice = int(rng.choice(pos_idx))
                        neg_choice = int(rng.choice(neg_idx))
                        selected = [pos_choice, neg_choice]
                        remaining = [i for i in range(len(train)) if i not in set(selected)]
                        if k > 2 and remaining:
                            fill = rng.choice(remaining, size=min(k - 2, len(remaining)), replace=False).tolist()
                            selected.extend(fill)
                else:
                    selected = rng.choice(len(train), size=k, replace=False).tolist()

            elif args.use_existing_reasoning:
                if scorer is None or tok is None:
                    raise ValueError("use_existing_reasoning requires scorer+tokenizer, but mode is rollout_only.")
                selected = _select_hard_examples(
                    train_rows=train,
                    reasoning_map=reasoning_map,
                    scorer=scorer,
                    tokenizer=tok,
                    device=device,
                    max_length=args.score_max_length,
                    batch_size=args.score_batch_size,
                    apply_sigmoid=args.score_sigmoid,
                    hard_topk=args.hard_topk,
                    hard_ratio=args.hard_ratio,
                    random_topk=args.random_topk,
                    random_ratio=args.random_ratio,
                )

            else:
                if scorer is None or tok is None:
                    raise ValueError("This selection branch requires scorer. Use --random_subset or switch mode.")
                base_texts = []
                for ex in train:
                    pid = ex.get("id")
                    reasoning_text = None
                    if reasoning_map is not None and pid is not None:
                        reasoning_text = reasoning_map.get(str(pid))
                    base_texts.append(_build_scorer_text(ex, reasoning_text=reasoning_text))

                base_scores = _score_texts(
                    scorer,
                    tok,
                    base_texts,
                    device=device,
                    max_length=args.score_max_length,
                    batch_size=args.score_batch_size,
                    apply_sigmoid=args.score_sigmoid,
                )

                indexed_scores = list(enumerate(base_scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)

                pos_idx, neg_idx = [], []
                for i, ex in enumerate(train):
                    if _get_label(ex) == 1:
                        pos_idx.append(i)
                    else:
                        neg_idx.append(i)

                if args.select_threshold is not None:
                    selected_pos = [i for i in pos_idx if base_scores>= args.select_threshold]
                    selected_neg = [i for i in neg_idx if base_scores<= args.select_threshold]
                    selected = selected_pos + selected_neg
                else:
                    if args.select_topk is not None:
                        k_total = max(1, int(args.select_topk))
                    elif args.select_ratio is not None:
                        k_total = max(1, int(len(train) * float(args.select_ratio)))
                    else:
                        k_total = max(1, int(len(train) * 0.2))
                    k_total = min(k_total, len(train))

                    if pos_idx and neg_idx:
                        if k_total < 2:
                            k_total = 2
                        k_pos = int(k_total * (len(pos_idx) / max(1, len(train))))
                        k_pos = max(1, min(len(pos_idx), k_pos))
                        k_neg = max(1, min(len(neg_idx), k_total - k_pos))
                        pos_sorted = sorted(pos_idx, key=lambda i: base_scores[i], reverse=True)
                        neg_sorted = sorted(neg_idx, key=lambda i: base_scores[i])
                        selected = pos_sorted[:k_pos] + neg_sorted[:k_neg]
                        if len(selected) < k_total:
                            pos_rem = [i for i in pos_sorted if i not in set(selected)]
                            neg_rem = [i for i in neg_sorted if i not in set(selected)]
                            fill = pos_rem + neg_rem
                            selected.extend(fill[: max(0, k_total - len(selected))])
                    else:
                        selected = [i for i, _ in indexed_scores[:k_total]]

                if not selected:
                    selected = [indexed_scores[0][0]]

            selected = _ensure_pos_neg_selected(selected, train, seed=int(args.seed) + int(r))

            _print_label_stats(
                f"\t(Round {r+1}) selected training subset's label stats:",
                [train[i] for i in selected]
            )

            selection_out = out_dir / f"phase3_train_selection_{round_tag}.jsonl"
            selected_set = set(selected)
            with open(selection_out, "w", encoding="utf-8") as f:
                if indexed_scores is not None:
                    for idx, score in indexed_scores:
                        ex = train[idx]
                        pid = ex.get("id", idx)
                        f.write(json.dumps({"id": pid, "score": score, "selected": idx in selected_set},
                                        ensure_ascii=False) + "\n")
                else:
                    for idx in selected:
                        ex = train[idx]
                        pid = ex.get("id", idx)
                        f.write(json.dumps({"id": pid, "score": None, "selected": True},
                                        ensure_ascii=False) + "\n")
            print(f"\t(Round {r+1}) save file:", selection_out)

            print(f"\n\tRollout (round {r+1})...")
            best_reasonings = []
            rollout_rows = []
            subset_rows = []

            for idx in tqdm(selected, desc="\trollouts", ncols=80, dynamic_ncols=False):
                ex = train[idx]
                pid = ex.get("id", ex.get("person_id", idx))
                lab = ex.get("label")
                label = int(lab.item()) if hasattr(lab, "item") else int(lab)

                age = ex.get("age")
                if age is not None:
                    age = age - 5
                sex = ex.get("sex", "unknown")

                prompt = clinical_reasoning_prompt(
                    ex.get("base_codes_diagnosis", []),
                    ex.get("base_codes_medication", []),
                    # ex.get("delta_codes_diagnosis", []),
                    # ex.get("delta_codes_medication", []),
                    memory,
                    sex,
                    age,
                )

                candidates = []
                for _ in range(int(args.rollout_n)):
                    rc_text = _generate_reasoning(
                        gen,
                        gen_tok,
                        args.gen_model_id,
                        prompt,
                        max_new_tokens=int(args.max_new_tokens_reasoning),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p) if args.top_p is not None else None,
                    )
                    candidates.append(rc_text)

                if not need_scorer:
                    for r_i, rc_text in enumerate(candidates):
                        rollout_rows.append(
                            {"id": pid, "label": label, "rollout": r_i, "score": None, "reasoning": rc_text}
                        )
                    subset_rows.append(ex)
                    continue

                cand_texts = [_build_scorer_text(ex, reasoning_text=rc) for rc in candidates]
                cand_scores = _score_texts(
                    scorer, tok, cand_texts,
                    device=device,
                    max_length=args.score_max_length,
                    batch_size=args.score_batch_size,
                    apply_sigmoid=args.score_sigmoid,
                )

                best_idx = int(np.argmax(cand_scores)) if label == 1 else int(np.argmin(cand_scores))
                best_reasoning = candidates[best_idx]
                best_score = float(cand_scores[best_idx])

                best_reasonings.append(
                    {"id": pid, "label": label, "reasoning": best_reasoning, "score": best_score, "rollout_n": int(best_idx)}
                )
                subset_rows.append(ex)

                for r_i, (rc_text, sc) in enumerate(zip(candidates, cand_scores)):
                    rollout_rows.append(
                        {"id": pid, "label": label, "rollout": r_i, "score": float(sc), "reasoning": rc_text}
                    )

            subset_out = out_dir / f"phase3_train_selection_data_{round_tag}.jsonl"
            rollout_out = out_dir / f"phase3_rollout_{round_tag}.jsonl"
            write_jsonl(subset_rows, str(subset_out))
            write_jsonl(rollout_rows, str(rollout_out))
            print(f"\t(Round {r+1}) save subset train jsonl:", subset_out)
            print(f"\t(Round {r+1}) save rollout jsonl:", rollout_out)

            reasoning_out = out_dir / f"phase3_rollout_best_{round_tag}.jsonl"
            if need_scorer:
                write_jsonl(best_reasonings, str(reasoning_out))
                print(f"\t(Round {r+1}) save selected reasoning jsonl:", reasoning_out)

        elif args.mode == "score_only":
            print("\n\tRescore existing rollout...")
            prerollpath = Path(args.out_dir) / args.preroll_path
            if not prerollpath.exists():
                raise ValueError(f'No preroll path: {prerollpath}')

            in_rollout = prerollpath / "phase3_rollout_r1.jsonl"
            in_subset = prerollpath / "phase3_train_selection_data_r1.jsonl"

            rollout_rows_in = read_jsonl(str(in_rollout))
            print(f"\tload rollout jsonl: {in_rollout} | n={len(rollout_rows_in)}")

            tok_path = _resolve_tokenizer_path(args.scorer_lora_dir, args.scorer_model_name)
            tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"

            scorer = _load_scorer_for_scoring(current_scorer_path, base_model_name, scorer_dtype)
            scorer.to(device)
            scorer.eval()
            scorer.config.pad_token_id = tok.pad_token_id

            by_pid = {}
            for row in rollout_rows_in:
                pid = row.get("id")
                if pid is None:
                    continue
                by_pid.setdefault(str(pid), []).append(row)

            best_reasonings = []
            rescored_rollout_rows = []

            for pid, rows in tqdm(by_pid.items(), desc="\trescoring", ncols=80, dynamic_ncols=False):
                ex = train_by_id.get(str(pid))
                if ex is None:
                    print('Skip ex due to no train_by_id returns None', pid)
                    continue
                label = _get_label(ex)

                cand_texts = []
                cand_reasonings = []
                cand_rollout_idx = []
                for rr in rows:
                    rc = rr.get("reasoning")
                    if rc is None:
                        print(f'Skip row due to no reasoning for pid {pid} found in this row' , rr)
                        continue
                    cand_reasonings.append(rc)
                    cand_rollout_idx.append(rr.get("rollout"))
                    cand_texts.append(_build_scorer_text(ex, reasoning_text=rc))

                if not cand_texts:
                    continue

                cand_scores = _score_texts(
                    scorer, tok, cand_texts,
                    device=device,
                    max_length=args.score_max_length,
                    batch_size=args.score_batch_size,
                    apply_sigmoid=args.score_sigmoid,
                )

                best_idx = int(np.argmax(cand_scores)) if label == 1 else int(np.argmin(cand_scores))
                best_reasonings.append(
                    {
                        "id": pid,
                        "label": int(label),
                        "reasoning": cand_reasonings[best_idx],
                        "score": float(cand_scores[best_idx]),
                        "rollout_n": int(cand_rollout_idx[best_idx]) if cand_rollout_idx[best_idx] is not None else int(best_idx),
                    }
                )

                for rc, sc, ridx in zip(cand_reasonings, cand_scores, cand_rollout_idx):
                    rescored_rollout_rows.append(
                        {"id": pid, "label": int(label), "rollout": ridx, "score": float(sc), "reasoning": rc}
                    )

            subset_out = out_dir / f"phase3_train_selection_data_{round_tag}.jsonl"
            reasoning_out = out_dir / f"phase3_rollout_best_{round_tag}.jsonl"
            rollout_out = out_dir / f"phase3_rollout_{round_tag}.jsonl"

            if in_subset.exists():
                subset_rows = read_jsonl(str(in_subset))
            else:
                subset_rows = [train_by_id[pid] for pid in by_pid.keys() if pid in train_by_id]

            write_jsonl(subset_rows, str(subset_out))
            write_jsonl(best_reasonings, str(reasoning_out))
            write_jsonl(rescored_rollout_rows, str(rollout_out))
            print(f"\tsave subset train jsonl:", subset_out)
            print(f"\tsave best reasoning jsonl:", reasoning_out)
            print(f"\tsave rescored rollout jsonl:", rollout_out)

        else:

            if args.mode != "preroll":
                print('No matched mode. Exit.')
                sys.exit()
            print("\n\tUse prerollout files and skip rollout generation...")
            prerollpath = Path(args.out_dir) / args.preroll_path
            if not prerollpath.exists():
                raise ValueError('No preroll path, error!')
            subset_out = prerollpath / f"phase3_train_selection_data_r1.jsonl"
            reasoning_out = prerollpath / f"phase3_rollout_best_r1.jsonl"
            rollout_out = prerollpath / f"phase3_rollout_r1.jsonl"
            best_reasonings = read_jsonl(str(reasoning_out))
            print(f'\t(Round {r+1}) load subset train jsonl:', subset_out)
            print(f'\t(Round {r+1}) load selected reasoning jsonl:', reasoning_out)
            print(f'\t(Round {r+1}) load rollout jsonl:', rollout_out)


        if args.mode in ["rollout_only"]:
            print("\trollout_only finished. Exit.")
            sys.exit()

        if args.mode in ["rollout_and_score", "score_only"]:
            print("\tbest generation finished (no training). Exit.")
            sys.exit()

        if args.train_scorer:
            train_output_dir = Path(args.train_output_dir) / run_id / round_tag
            _train_scorer_from_subset(
                subset_jsonl=subset_out,
                reasoning_rows=best_reasonings,
                eval_reasoning_map=reasoning_map,
                output_dir=str(train_output_dir),
                model_path=train_model_path,
                base_model_name=base_model_name,
                dtype=args.train_dtype,
                seed=args.seed,
                lr=args.train_lr,
                epochs=args.train_epochs,
                batch_size=args.train_batch_size,
                grad_accum=args.train_grad_accum,
                max_length=args.train_max_length,
                neg_per_pos=args.train_neg_per_pos,
                save_steps=args.train_save_steps,
                score_only=args.train_scorer_only,
                lora_last_n=args.train_lora_last_n,
                loss_type=args.train_loss_type,
                pointwise_alpha=args.train_pointwise_alpha,
                pairwise_margin=args.train_pairwise_margin,
                pos_weight=args.train_pos_weight,
                margin=args.train_margin,
                margin_on_sigmoid=args.train_margin_on_sigmoid,
                device=device,
                eval_jsonl=eval_jsonl,
                eval_steps=args.eval_steps,
                eval_batch_size=args.eval_batch_size,
                eval_sigmoid=args.eval_sigmoid,
            )
            current_scorer_path = str(train_output_dir)
            train_model_path = current_scorer_path

if __name__ == "__main__":
    main()
