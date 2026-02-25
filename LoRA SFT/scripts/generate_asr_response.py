#!/usr/bin/env python3
"""
Generate a postprocessed transcript from an ASR text file using the fine-tuned LoRA adapter.

Defaults assume you trained with run_name=random, which saved into outputs/qwen2p5-14b-random.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Optional

# Import Unsloth first for proper patching/optimizations
from unsloth import FastLanguageModel  # noqa: E402
from peft import PeftModel  # noqa: E402
import torch  # noqa: E402


def load_prompt_template(module_path: str) -> str:
    spec = importlib.util.spec_from_file_location("format_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    if not hasattr(mod, "PROMPT_TEMPLATE"):
        raise ValueError(f"{module_path} must define PROMPT_TEMPLATE")
    template = getattr(mod, "PROMPT_TEMPLATE")
    if not isinstance(template, str):
        raise TypeError("PROMPT_TEMPLATE must be a string")
    return template


def detect_prompt_style(template: str) -> str:
    """Return 'alpaca' if the template looks like Alpaca (has '### Input:'),
    otherwise 'plain'."""
    if "### Input:" in template:
        return "alpaca"
    return "plain"


def build_prompt(template: str, asr_text: str) -> str:
    style = detect_prompt_style(template)
    if style == "alpaca":
        # Match the training format used in train_qwen.py (Alpaca-style)
        return f"{template}{asr_text}\n\n### Response:\n"
    # Fallback to the simpler plain prompt style
    return f"{template}{asr_text}\n\n# Output\n"


def generate(
    adapter_dir: str,
    base_model: str,
    prompt: str,
    max_seq_length: int = 4096,
    max_new_tokens: int = 1500,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device_map: Optional[str] = "auto",
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    # Load base model and attach LoRA adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=False,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # Tokenize input
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    for k in enc:
        enc[k] = enc[k].to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    if repetition_penalty and repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Extract only the completion after the prompt
    if text.startswith(prompt):
        completion = text[len(prompt):].strip()
    else:
        # Try both markers depending on style
        for marker in ("\n### Response:\n", "\n# Output\n"):
            if marker in text:
                completion = text.split(marker, 1)[-1].strip()
                break
        else:
            completion = text
    return completion


def main():
    ap = argparse.ArgumentParser(description="Generate ASR postprocessed transcript with fine-tuned adapter")
    ap.add_argument("--input", default="aprocsa1944a Azure.txt", help="Path to ASR text file")
    ap.add_argument("--prompt_module", default="format.py", help="Path to Python file exposing PROMPT_TEMPLATE")
    ap.add_argument("--adapter_dir", default=os.path.join("outputs", "qwen2p5-14b-random"), help="LoRA adapter output dir")
    ap.add_argument("--base_model", default="unsloth/Qwen2.5-14B", help="Base model to load")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=1800, help="Max tokens to generate (set high for long outputs)")
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help=">1.0 to discourage repeats (e.g., 1.05–1.2)")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Disallow repeating n-grams of this size (e.g., 4)")
    ap.add_argument("--output", default=None, help="Where to save the result; defaults to <adapter_dir>/<input_basename>_response.txt")
    args = ap.parse_args()

    template = load_prompt_template(args.prompt_module)
    with open(args.input, "r", encoding="utf-8") as f:
        asr_text = f.read().strip()

    prompt = build_prompt(template, asr_text)
    completion = generate(
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        prompt=prompt,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    os.makedirs(args.adapter_dir, exist_ok=True)
    out_path = args.output or os.path.join(
        args.adapter_dir,
        f"{os.path.splitext(os.path.basename(args.input))[0]}_response.txt",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(completion)

    print(f"Saved response to: {out_path}")
    print("--- Preview (first 600 chars) ---")
    print(completion[:600])


if __name__ == "__main__":
    main()
