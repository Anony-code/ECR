from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def qwen_init(
    device_map: str = "auto",
    max_memory: Optional[Dict[int, str]] = None,
    dtype: str = "auto",
):
    model_name = "Qwen/Qwen3-4B-Instruct-2507"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {
        "dtype": dtype,
        "device_map": device_map,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


def qwen_generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> str:
    """
    Unified Qwen text generation utility.
    - Handles chat template
    - Handles prompt slicing
    - Returns ONLY assistant text

    Args:
        model: AutoModelForCausalLM (Qwen)
        tokenizer: Qwen tokenizer
        messages: [{"role": "system"/"user", "content": "..."}]
        max_new_tokens: generation length
        do_sample: sampling or greedy
        temperature: temperature (used only if do_sample=True)
        top_p: nucleus sampling (optional)

    Returns:
        str: assistant-generated text
    """

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    input_len = model_inputs.input_ids.shape[-1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            **gen_kwargs,
        )

    output_ids = generated_ids[0][input_len:]

    output_text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )

    return output_text.strip()
