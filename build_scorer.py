import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import IterableDataset, Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback, DataCollatorWithPadding
from trl import RewardTrainer, RewardConfig

from utils import get_evaluation_metrics
import sys
import os
import random
import warnings
import torch.nn as nn
from scorer_prompts import BASELINE_PROMPT_AD, BASELINE_PROMPT_ADRD, BASELINE_PROMPT_PD



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


def replace_score_with_mlp(model, hidden=None, mid=256, dropout=0.1):
    """
    Replace model.score (Linear) with a 2-layer MLP: hidden -> mid -> 1.
    Ensures dtype/device match the base model.
    """
    if hidden is None:
        hidden = int(getattr(model.config, "hidden_size", 1024))

    mlp = nn.Sequential(
        nn.Linear(hidden, mid),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid, 1),
    )

    # Make sure new head matches model dtype/device (important!)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mlp = mlp.to(device=device, dtype=dtype)

    model.score = mlp
    model.config.num_labels = 1
    return model

 
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
            center_loss = self.args.center_rewards_coefficient * torch.mean((chosen_scores + rejected_scores) ** 2)

        # Keep RewardTrainer-style metrics for logging/monitoring.
        if hasattr(self, "_metrics") and hasattr(self, "accelerator") and mode in self._metrics:
            with torch.no_grad():
                all_rewards = self.accelerator.gather(torch.cat([raw_chosen_scores, raw_rejected_scores], dim=0))
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


        # TRL-style pairwise loss with optional margin: -log(sigmoid((chosen - rejected) - margin)).
        diff = chosen_scores - rejected_scores - self.pairwise_margin
        pairwise_loss = -torch.nn.functional.logsigmoid(diff).mean()
        if self.loss_type == "pairwise":
            if return_outputs:
                pairwise_loss += center_loss
                return pairwise_loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
            pairwise_loss += center_loss
            return pairwise_loss

        # Pointwise BCE on chosen=1, rejected=0.
        pointwise_targets = torch.cat(
            [
                torch.ones_like(chosen_scores),
                torch.zeros_like(rejected_scores),
            ],
            dim=0,
        )
        pointwise_logits = torch.cat([chosen_scores, rejected_scores], dim=0)
        pointwise_loss = self.pointwise_bce(pointwise_logits, pointwise_targets)
        if self.loss_type == 'bce':
            loss = pointwise_loss + center_loss
        else:
            assert self.loss_type =='pairwise_bce'
            loss = pairwise_loss + self.pointwise_alpha * pointwise_loss + center_loss

        if return_outputs:
            return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
        return loss


def _reset_score_head(model, seed):
    if not hasattr(model, "score"):
        return False
    score = model.score
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if hasattr(score, "reset_parameters"):
            score.reset_parameters()
        else:
            for module in score.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
    return True


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_jsonl_multi(paths):
    if isinstance(paths, (list, tuple)):
        rows = []
        for p in paths:
            rows.extend(_read_jsonl(str(p)))
        return rows
    return _read_jsonl(str(paths))


def _find_jsonl_files(data_dir: Path, pattern: str):
    data_dir = Path(data_dir)
    matched = sorted(data_dir.glob(pattern))
    if not matched:
        raise FileNotFoundError(f"No files found with pattern {pattern} in {data_dir}")
    return matched


def _format_baseline_text(ex, reasoning_text=None):
    base_dx = ex.get("base_codes_diagnosis", []) or []
    base_md = ex.get("base_codes_medication", []) or []
    # delta_dx = ex.get("delta_codes_diagnosis", []) or []
    # delta_md = ex.get("delta_codes_medication", []) or []

    base_dx = [i.lower().strip() for i in base_dx]
    base_md = [i.lower().strip() for i in base_md]
    # delta_dx = [i.lower().strip() for i in delta_dx]
    # delta_md = [i.lower().strip() for i in delta_md]
    parts = [
        f"Age: {ex.get('age') - 5};"
        f"Sex: {ex.get('sex')};"
        f"Baseline diagnoses: {'; '.join(base_dx)}",
        f"Baseline medications: {'; '.join(base_md)}",
    ]
    if reasoning_text:
        parts.append('Analysis for this patient: ' + str(reasoning_text))

    return "\n".join(parts)


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

    print(
        f"[Info] train dataset: cases={len(positives)}, controls={len(negatives)}, "
        f"total={len(positives) + len(negatives)}, with_reasoning={have_reasoning}"
    )
    if not positives or not negatives:
        raise ValueError(f"Need both pos and neg. Got pos={len(positives)}, neg={len(negatives)}")

    state = EpochState(base_seed=base_seed)

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
                if include_meta:
                    out["meta"] = {"pid_chosen": pos["pid"], "pid_rejected": neg["pid"]}
                yield out

    ds = IterableDataset.from_generator(gen)
    return ds, state, len(positives)


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
        f"[Info] test dataset: cases={len(positives)}, controls={len(negatives)}, "
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
    raw_scores_all = []
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
                # enc = tokenizer.pad(features, padding="longest", return_tensors="pt")
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

            raw_scores_all.append(scores.detach().float().cpu())
            if apply_sigmoid:
                scores = torch.sigmoid(scores)

            scores_all.append(scores.detach().float().cpu())
            if isinstance(labels, torch.Tensor):
                labels_all.append(labels.detach().to(dtype=torch.int64).cpu())
            else:
                labels_all.append(torch.tensor(labels, dtype=torch.int64))

    raw_scores = torch.cat(raw_scores_all).numpy()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dy_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, default=None)
    ap.add_argument("--test_jsonl", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="model_ckpt_baseline")
    ap.add_argument("--max_length", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--neg_per_pos", type=int, default=2)
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--eval_sigmoid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save_steps", type=int, default=0)
    ap.add_argument("--runnote", type=str, default="n")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval_only", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--score_only", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--pointwise_alpha", type=float, default=0.2)
    ap.add_argument("--pairwise_margin", type=float, default=0.0)
    ap.add_argument("--pos_weight", type=float, default=None)
    ap.add_argument("--reasoning_jsonl", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument("--reasoning_pid_key", type=str, default="id")
    ap.add_argument("--reasoning_text_key", type=str, default="reasoning")
    ap.add_argument("--dataset", type=str, default="ad")

    ap.add_argument(
        "--loss_type",
        type=str,
        default="pairwise_bce",
        choices=["pairwise", "pairwise_bce", "margin", "bce"],
    )

    ap.add_argument(
        "--lora_last_n",
        type=int,
        default=1,
        help="Apply LoRA only to the last N transformer blocks. 0 or negative means all layers.",
    )

    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument(
        "--margin_on_sigmoid",
        action="store_true",
        default=True,
        help="Apply sigmoid to each score before margin (default).",
    )
    ap.add_argument(
        "--margin_on_logit",
        action="store_false",
        dest="margin_on_sigmoid",
        help="Use raw logits for margin instead of sigmoid.",
    )

    args = ap.parse_args()
    print(f"[Info] CUDA devices: visible={os.environ.get('CUDA_VISIBLE_DEVICES')}, count={torch.cuda.device_count()}")

    global BASELINE_PROMPT
    if args.dataset =='ad':
        BASELINE_PROMPT = BASELINE_PROMPT_AD
    if args.dataset =='adrd':
        BASELINE_PROMPT = BASELINE_PROMPT_ADRD
    if args.dataset =='pd':
        BASELINE_PROMPT = BASELINE_PROMPT_PD

    print(f"[Info] dataset={args.dataset}")
    print("[Info] arguments:", args)

    output_dir = Path("artifacts", "phase2", args.output_dir )
    if args.runnote is not None:
        output_dir = output_dir / args.runnote
    if args.ckpt_dir:
        output_dir = output_dir / args.ckpt_dir

    data_dir = Path("data")
    dy_jsonl = data_dir / args.dy_jsonl
    if args.test_jsonl is not None:
        test_jsonl = data_dir / args.test_jsonl
    else:
        test_jsonl = None

    print(f"[Info] use reasoning: {args.reasoning_jsonl}")
    reasoning_map = {}
    if args.reasoning_jsonl:

        if args.dataset == 'ad':
            ratio = 10
            train_jsonl_reasoning = f'artifacts/phase1/{args.dataset}/train_set_{ratio}_reasoning.jsonl'
            train_reasoning_rows = _read_jsonl(train_jsonl_reasoning)

            test_jsonl_reasoning = _find_jsonl_files(
                data_dir=f"artifacts/phase1/{args.dataset}",
                pattern="test_set_reasoning",
            )
            test_reasoning_rows = _read_jsonl_multi(test_jsonl_reasoning)


        elif args.dataset == 'pd':
            ratio = 5
            train_jsonl_reasoning = f'artifacts/phase1/{args.dataset}/train_set_{ratio}_reasoning.jsonl'
            train_reasoning_rows = _read_jsonl(train_jsonl_reasoning)

            test_jsonl_reasoning = _find_jsonl_files(
                data_dir=f"artifacts/phase1/{args.dataset}",
                pattern="test_set_reasoning",
            )
            test_reasoning_rows = _read_jsonl_multi(test_jsonl_reasoning)



        elif args.dataset == 'adrd':
            ratio = 10
 
            train_jsonl_reasoning = f'artifacts/phase1/{args.dataset}/train_set_{ratio}_reasoning.json'
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


    if args.eval_only:
        if test_jsonl is None:
            raise ValueError("--eval_only requires --test_jsonl")
        tok = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=1,
            dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, output_dir)
        model.to(device)
        model.config.pad_token_id = tok.pad_token_id
        test_ds = build_pointwise_dataset_baseline(
            test_jsonl,
            reasoning_map=reasoning_map,
        )
        scores, labels = eval_pointwise(
            model,
            tok,
            test_ds,
            text_key="text",
            label_key="label",
            batch_size=args.eval_batch_size,
            max_length=args.max_length,
            apply_sigmoid=args.eval_sigmoid,
        )
        auroc, auprc, f1, sensitivity_90, sensitivity_95, ppv_90, ppv_95 = (
            get_evaluation_metrics(labels, scores)
        )
        print(
            f"[EvalOnly] AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"F1={f1:.4f} Sens@90%={sensitivity_90:.4f} Sens@95%={sensitivity_95:.4f} "
            f"PPV@90%={ppv_90:.4f} PPV@95%={ppv_95:.4f}"
        )
        return

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        dtype=torch.bfloat16,
    )
    _reset_score_head(model, args.seed)
    model.to(device)
    print(f"[Info] model ready on device: {next(model.parameters()).device}")

    dy_ds, epoch_state, pos_count = build_pref_iterable_dataset_epoch_baseline(
        in_jsonl=str(dy_jsonl),
        neg_per_pos=args.neg_per_pos,
        base_seed=args.seed,
        prompt=None,
        include_meta=False,
        reasoning_map=reasoning_map,
    )

    pairs_per_epoch = pos_count * args.neg_per_pos
    bs = args.batch_size
    ga = args.grad_accum
    steps_per_epoch = math.ceil(pairs_per_epoch / (bs * ga))
    max_steps = steps_per_epoch * args.epochs
    print(
        f"[Info] training setup: pos_count={pos_count}, neg_per_pos={args.neg_per_pos}, "
        f"pairs_per_epoch={pairs_per_epoch}, batch_size={bs}, grad_accum={ga}, "
        f"steps_per_epoch={steps_per_epoch}, max_steps={max_steps}"
    )

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.config.pad_token_id = tok.pad_token_id

    print(f"[Info] tokenizer ready: padding_side={tok.padding_side}, pad_token_id={tok.pad_token_id}")

    rcfg = RewardConfig(
        output_dir=output_dir,
        learning_rate=args.lr,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        logging_steps=100,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=max(args.save_steps, 1),
        eval_strategy="steps" if args.eval_jsonl else "no",
        eval_steps=1 if args.eval_jsonl else None,
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
    if args.score_only:
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "score" in name:
                p.requires_grad = True
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        warnings.filterwarnings(
            "ignore",
            message="None of the inputs have requires_grad=True. Gradients will be None",
            category=UserWarning,
        )
    else:
        if args.lora_last_n > 0:
            layers, layers_pattern = _resolve_transformer_layers(model)
            if layers is None:
                raise ValueError(
                    "Unable to locate transformer layers; cannot apply --lora_last_n. "
                    "Please specify a model with standard layer attributes."
                )
            num_layers = len(layers)
            n_last = min(args.lora_last_n, num_layers)
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
            print(f"\n===== LoRA last-N layers =====")
            print(f"layers_pattern: {layers_pattern}")
            print(f"total_layers: {num_layers}")
            print(f"lora_last_n: {n_last}")
        peft_config = lora


    trainer = LossSwitchRewardTrainer(
        model=model,
        args=rcfg,
        train_dataset=dy_ds,
        processing_class=tok,
        peft_config=peft_config,
        loss_type=args.loss_type,
        pointwise_alpha=args.pointwise_alpha,
        pairwise_margin=args.pairwise_margin,
        pos_weight=args.pos_weight,
        margin=args.margin,
        margin_on_sigmoid=args.margin_on_sigmoid,
    )
    if (args.test_jsonl is not None) and (args.eval_steps > 0):
        print("\n===== Add test callback =====")
        eval_callback = PeriodicEvalCallback(
            test_jsonl=test_jsonl,
            tokenizer=tok,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            eval_every_steps=args.eval_steps,
            apply_sigmoid=args.eval_sigmoid,
            reasoning_map=reasoning_map,
        )
        trainer.add_callback(eval_callback)

    print("\n===== Trainable parameters =====")
    if args.score_only:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Score only: trainable params: {trainable} || all params: {total} || trainable%: {trainable / total * 100:.4f}")
    else:
        trainer.model.print_trainable_parameters()
        print("LoRA modules_to_save:", lora.modules_to_save)
        print("LoRA target_modules:", lora.target_modules)

    print("\n===== Start training =====")
    trainer.train()

    if args.score_only:
        print("[Info] saving full model (score_only mode)")
        trainer.model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
    else:
        print("[Info] saving PEFT model (LoRA mode)")
        if not isinstance(trainer.model, PeftModel):
            raise ValueError("Expected PeftModel for LoRA saving.")
        trainer.model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)


    print("\n===== Final Evaluation =====")
    if (args.test_jsonl is not None) and (args.eval_steps > 0):
        eval_callback.single_eval(model, trainer.state.global_step)


if __name__ == "__main__":
    main()
