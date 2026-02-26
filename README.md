
This project runs in the following steps:

1. Prepare training/testing JSONL data
2. Build memory with an LLM
3. Train a scorer model
4. Run EM-style interaction (rollout + scoring)

# Requirement
torch
transformers>=4.43.0
accelerate
safetensors
tokenizers
datasets
peft
trl
scikit-learn
numpy
pyyaml

# Datasets
We evaluate the proposed model using real-world healthcare data and leverage the national All of Us Research Platform to construct the AD and PD cohorts, and employ one reginal cohort from the OHSU EHR data warehouse

# Run

## 1) Prepare data

```bash
python prepare_data.py --dataset ad --fold 0 --ratio 10 --use_neg_ratio --neg_ratio 5
```

## 2) Build memory

First, update `configs/gen_memory_qwen_ad.yaml` so `train_jsonl` points to the training file generated in step 1.

```bash
CUDA_VISIBLE_DEVICES=0 python build_memory.py \
  --config configs/gen_memory_qwen_ad.yaml \
  --device_map cuda:0 \
  --runnote demo
```

## 3) Build scorer

```bash
CUDA_VISIBLE_DEVICES=0 python build_scorer.py \
  --dataset ad \
  --dy_jsonl AD_train_fold0_ratio10_neg5_visit2.jsonl \
  --test_jsonl AD_test_fold0_ratio10_neg5_visit2.jsonl \
  --output_dir scorer_ad \
  --runnote run1
```

## 4) Run interaction (rollout + score)

```bash
CUDA_VISIBLE_DEVICES=0 python em_interaction_modes.py \
  --mode rollout_and_score \
  --dataset ad \
  --train_jsonl data/AD_train_fold0_ratio10_neg5_visit2.jsonl \
  --memory_path artifacts/phase1/ad/ratio10_memory_final_ad.json \
  --scorer_lora_dir artifacts/phase2/scorer_ad/run1 \
  --runnote em1
```

 

## Update
The code will be further organized and refactored upon acceptance.