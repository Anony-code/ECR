
This project runs in the following steps:

1. Prepare training/testing JSONL data
2. Build memory with an LLM
3. Train a scorer model
4. Run EM-style interaction (rollout + scoring)

# Requirement
+ torch
+ transformers>=4.43.0
+ accelerate
+ safetensors
+ tokenizers
+ datasets
+ peft
+ trl
+ scikit-learn
+ numpy
+ pyyaml

# Datasets
We evaluate the proposed model using real-world healthcare data and leverage the national [All of Us Research Platform](https://www.researchallofus.org/) to construct the AD and PD cohorts, and employ one regional cohort from the [OHSU EHR data warehouse](https://research-data-catalog.ohsu.edu/records/ksqgw-95972).

# Run

## 1) Prepare data

```bash
python prepare_data.py --dataset ad   --ratio 10 
```

## 2) Build memory

First, update `configs/gen_memory_qwen_ad.yaml` so `train_jsonl` points to the training file generated in step 1.

```bash
CUDA_VISIBLE_DEVICES=0 python build_memory.py \
  --config configs/gen_memory_qwen_ad.yaml 
```

## 3) Build scorer

```bash
CUDA_VISIBLE_DEVICES=0 python build_scorer.py \
  --dataset ad \
  --dy_jsonl data/AD_train_ratio10.jsonl \
  --test_jsonl data/AD_test_ratio10.jsonl \
  --output_dir scorer_ad \
  --runnote ... 
```

## 4) Run interaction (rollout + score)

```bash
CUDA_VISIBLE_DEVICES=0 python em_interaction_modes.py \
  --mode preroll \
  --dataset ad \
  --train_jsonl data/AD_train_ratio10.jsonl  \
  --memory_path ... \
  --scorer_lora_dir ... \
  --runnote ... 
```

 

## Update
The code will be further organized and refactored upon acceptance.
