from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import yaml
from transformers import set_seed
from tqdm import tqdm
from transformers import pipeline, set_seed
from utils import read_jsonl, save_json, extract_json, qwen_generate, qwen_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--runnote", type=str, default="n")

    args = ap.parse_args()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    rand = random.randint(10, 99)
    run_id = f"{timestamp}_{rand}"
    run_note = args.runnote + "_" + run_id
    print("\nStart the new run with id", run_note)

    cfg = yaml.safe_load(open(args.config, "r"))
    set_seed(int(cfg.get("seed", 7)))

    os.makedirs("artifacts", exist_ok=True)
    if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
        model_name = "qwen-4b-instruct"

    try:
        module = importlib.import_module(f"prompts.prompts_{args.dataset}")

        clinical_reasoning_prompt = module.clinical_reasoning_prompt
        calibration_update_prompt = module.calibration_update_prompt
        memory_refresh_prompt_qwen = module.memory_refresh_prompt_qwen
        memory_refresh_prompt = module.memory_refresh_prompt
        print("\nLoad prompt for dataset", args.dataset)

        from memory_schema import init_memory, init_memory_adrd, init_memory_pd
        INIT_MEMORY = {
            "adrd": init_memory_adrd,
            "pd": init_memory_pd,
            "ad": init_memory,
        }

        init_memory = INIT_MEMORY.get(args.dataset)

    except:
        raise ValueError(f"Unsupported cohort prompt: {args.dataset}")

    train = read_jsonl(cfg["train_jsonl"])
    print("\n[Info] training data loaded:", len(train))

    memory = init_memory()
    print("[Info] memory initialized.")

    max_templates = int(cfg["memory_budget"]["max_templates"])
    max_lessons = int(cfg["memory_budget"]["max_lessons"])
    print(f"[Info] memory budget - max_templates: {max_templates}, max_lessons: {max_lessons}")

    json_retries = int(cfg.get("json_retries", 2))
    print(f"[Info] json_retries set to: {json_retries}")

    # Refresh frequency and log paths.
    refresh_every = int(cfg.get("memory_refresh_every", 25))
    print(f"[Info] memory_refresh_every set to: {refresh_every}")

    log_path_reasoning = Path(cfg.get("phase1_text_log_reasoning"))
    log_path_reasoning = log_path_reasoning.with_name(
        f"{log_path_reasoning.stem}_{run_note}{log_path_reasoning.suffix}"
    )
    print(f"\n[Info] text log path (reasoning &): {log_path_reasoning}")

    log_path_calibration = Path(cfg.get("phase1_text_log_calibration"))
    log_path_calibration = log_path_calibration.with_name(
        f"{log_path_calibration.stem}_{run_note}{log_path_calibration.suffix}"
    )
    print(f"[Info] text log path (& calibration): {log_path_calibration}")

    memlog_path = Path(cfg.get("memory_refresh_log"))
    memlog_path = memlog_path.with_name(
        f"{memlog_path.stem}_{run_note}{memlog_path.suffix}"
    )
    print(f"[Info] memory refresh log path: {memlog_path}")

    memory_final_output = Path(cfg["memory_out"])
    memory_final_output = memory_final_output.with_name(
        f"{memory_final_output.stem}_{run_note}{memory_final_output.suffix}"
    )
    print(f"[Info] memory final output path: {memory_final_output}")
 
    # Load generator.
    if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
        gen, tokenizer = qwen_init(
            device_map=args.device_map,
            max_memory=cfg.get("max_memory"),
            dtype=cfg.get("dtype", "auto"),
        ) 

    print("\n[Info] loading generation model:", cfg["gen_model_id"])
    print("[Info] starting memory building...")

    calib_buffer = []
    reason_buffer = []
    for i, ex in enumerate(tqdm(train, total=len(train)), start=1):
        if not isinstance(ex, dict):
            print(f"[WARN!] bad example type i={i}: {type(ex)} -> skip")
            continue

        base_codes_diag = ex.get("base_codes_diagnosis", [])
        base_codes_med = ex.get("base_codes_medication", [])
        delta_codes_diag = ex.get("delta_codes_diagnosis", [])
        delta_codes_med = ex.get("delta_codes_medication", [])
        sex = ex.get("sex", "unknown")
        age = ex.get("age", None) - 5

        lab = ex["label"]
        label = int(lab.item()) if hasattr(lab, "item") else int(lab)

        ex_id = ex.get("person_id", ex.get("id", i))  # best-effort id

        # Load prompt for reasoning.
        rc_prompt = clinical_reasoning_prompt(
            base_codes_diag,
            base_codes_med,
            delta_codes_diag,
            delta_codes_med,
            memory,
            sex,
            age,
        )
 
        mess = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": rc_prompt},
        ]

        if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
            rc_text = qwen_generate(
                gen,
                tokenizer,
                mess,
                max_new_tokens=int(cfg["max_new_tokens_reasoning"]),
                do_sample=False,
                temperature=1,
            )
            reasoning_card = rc_text
 
        # Load prompt for calibration.
        calib_prompt = calibration_update_prompt(
            memory=memory,
            reasoning_card=reasoning_card,
            label=label,
        )
        calib_mess = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": calib_prompt},
        ]

        if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
            calib_text = qwen_generate(
                gen,
                tokenizer,
                calib_mess,
                max_new_tokens=int(cfg["max_new_tokens_calibration"]),
                do_sample=False,
                temperature=1,
            )
 
        with open(log_path_reasoning, "a") as f:
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "id": ex_id,
                        "label": label,
                        "reasoning": rc_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        with open(log_path_calibration, "a") as f:
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "id": ex_id,
                        "label": label,
                        "calibration": calib_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if calib_text:
            calib_buffer.append(calib_text)
            reason_buffer.append(rc_text)

        if i % refresh_every == 0 and calib_buffer:
            if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
                refresh_prompt = memory_refresh_prompt_qwen(
                    memory=memory,
                    new_calib_paragraphs=calib_buffer,
                    new_reason_paragraphs=reason_buffer,
                )
 
            refresh_mess = [
                {"role": "system", "content": "Reasoning: low"},
                {"role": "user", "content": refresh_prompt},
            ]

            if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
                refreshed_memory = qwen_generate(
                    gen,
                    tokenizer,
                    refresh_mess,
                    max_new_tokens=int(
                        cfg.get("max_new_tokens_memory_refresh", cfg["max_new_tokens_calibration"])
                    ),
                    do_sample=False,
                    temperature=1,
                )
                refreshed_memory_card = refreshed_memory.split("'experience_knowledge': ")[-1]
 

            if refreshed_memory is not None and len(refreshed_memory_card) > 0:
                old_mem = memory["experience_knowledge"]
                memory["experience_knowledge"] = refreshed_memory_card

                with open(memlog_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "i": i,
                                "buffer_size": len(calib_buffer),
                                "old_memory": old_mem,
                                "new_memory": refreshed_memory_card,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if i % 200 == 0:
                    print(f"[Info] refreshed @ i={i}; buffer_size={len(calib_buffer)}")
            calib_buffer = []  # reset buffer
            reason_buffer = []  # reset buffer

    if len(calib_buffer) > 0:
        print("[Info] final memory refresh at end of data at i=", i)
        if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
            refresh_prompt_leftover = memory_refresh_prompt_qwen(
                memory=memory,
                new_calib_paragraphs=calib_buffer,
                new_reason_paragraphs=reason_buffer,
            )
 
        refresh_leftover_mess = [
            {"role": "system", "content": "Reasoning: low"},
            {"role": "user", "content": refresh_prompt_leftover},
        ]

        if cfg["gen_model_id"] == "Qwen/Qwen3-4B-Instruct-2507":
            refreshed_memory_leftover = qwen_generate(
                gen,
                tokenizer,
                refresh_leftover_mess,
                max_new_tokens=int(
                    cfg.get("max_new_tokens_memory_refresh", cfg["max_new_tokens_calibration"])
                ),
                do_sample=False,
                temperature=1,
            )
            refreshed_memory_leftover_card = refreshed_memory_leftover.split("'experience_knowledge': ")[-1]
 
        if refreshed_memory_leftover is not None and len(refreshed_memory_leftover_card) > 0:
            old_mem = memory["experience_knowledge"]
            memory["experience_knowledge"] = refreshed_memory_leftover_card

            with open(memlog_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "i": i,
                            "buffer_size": len(calib_buffer),
                            "old_memory": old_mem,
                            "new_memory": refreshed_memory_leftover_card,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(f"[Info] leftover refreshed @ i={i}; buffer_size={len(calib_buffer)}")

    save_json(memory, memory_final_output)
    print(f"[Info] final memory saved: {memory_final_output}")


if __name__ == "__main__":
    main()
