import argparse
import json
import os
from datetime import datetime

from datasets import load_dataset


def build_alpaca_text(instruction: str, input_text: str, output_text: str) -> str:
    template = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n"
        "### Input:\n{}\n\n"
        "### Response:\n{}"
    )
    return template.format(instruction, input_text, output_text)


def load_and_prepare_dataset(path: str, tokenizer, eos_token: str, ignore_text_field: bool = False):
    ds = load_dataset("json", data_files=path, split="train")

    # Prefer preformatted `text` if present; ensure EOS at end
    if (not ignore_text_field) and ("text" in ds.column_names):
        def ensure_eos(example):
            t = example.get("text", "")
            if not isinstance(t, str):
                t = str(t)
            if not t.endswith(eos_token):
                t = t + eos_token
            return {"text": t}

        return ds.map(ensure_eos)

    # Otherwise, format from instruction/input/output
    required_cols = {"instruction", "input", "output"}
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {sorted(missing)} or preformatted 'text'."
        )

    def _map_fn(examples):
        texts = []
        for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
            texts.append(build_alpaca_text(inst, inp, out) + eos_token)
        return {"text": texts}

    return ds.map(_map_fn, batched=True, remove_columns=[c for c in ds.column_names if c != "text"])


def main():
    parser = argparse.ArgumentParser(description="SFT LoRA fine-tuning for Qwen2.5-14B with Unsloth")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--run_name", required=True, help="Run tag, e.g. random or exhaustive")
    parser.add_argument("--model_name", default="unsloth/Qwen2.5-14B", help="Base model name")
    parser.add_argument("--output_dir", default="outputs", help="Base output directory")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=2000, help="Total training steps; set 0 to use num_train_epochs")
    parser.add_argument("--num_train_epochs", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 if available")
    parser.add_argument("--merge_adapter", action="store_true", help="Save merged model weights as well")
    parser.add_argument("--packing", action="store_true", help="Enable TRL sequence packing")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--report_to", default="tensorboard", choices=["none", "tensorboard", "wandb"], help="Logging backend")
    parser.add_argument("--device_map", default=None, help="Device map for model sharding across GPUs (e.g. auto, balanced, sequential, or a JSON). Leave empty for DDP with accelerate.")
    parser.add_argument("--ignore_text_field", action="store_true", help="Ignore preformatted 'text' even if present; rebuild from instruction/input/output.")
    args = parser.parse_args()

    # Imports that require installed libs; provide helpful error if missing
    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except Exception as e:
        raise SystemExit(
            "Missing dependencies. Please install: 'pip install unsloth trl transformers peft bitsandbytes datasets accelerate'"
        ) from e

    # Configure model load
    load_in_4bit = not args.no_4bit
    dtype = None  # auto; Ampere+ will use bfloat16

    # Determine local rank/device for DDP
    import torch
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    from_pretrained_kwargs = dict(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    # If user specified a device_map, respect it. Otherwise, for DDP place the model on the local rank device.
    if args.device_map is not None and str(args.device_map).strip():
        from_pretrained_kwargs["device_map"] = args.device_map
    else:
        if torch.cuda.is_available():
            from_pretrained_kwargs["device_map"] = {"": local_rank}

    model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)

    # LoRA config
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    eos_token = tokenizer.eos_token or "<|endoftext|>"
    dataset = load_and_prepare_dataset(args.dataset, tokenizer, eos_token, ignore_text_field=args.ignore_text_field)

    # Resolve output dir
    base = os.path.abspath(args.output_dir)
    tag = args.run_name.strip().replace("/", "-")
    out_dir = os.path.join(base, f"qwen2p5-14b-{tag}")
    os.makedirs(out_dir, exist_ok=True)
    logging_dir = os.path.join(out_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    # Trainer
    training_args = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": out_dir,
        "report_to": args.report_to,
        "logging_dir": logging_dir,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "dataloader_num_workers": args.dataloader_num_workers,
        # Keep DDP stable; SFTConfig supports ddp_find_unused_parameters but not ddp_static_graph
        "ddp_find_unused_parameters": False,
    }
    if args.max_steps and args.max_steps > 0:
        training_args["max_steps"] = args.max_steps
    if args.num_train_epochs and args.num_train_epochs > 0:
        training_args["num_train_epochs"] = args.num_train_epochs
    if args.bf16:
        training_args["bf16"] = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        args=SFTConfig(**training_args),
    )

    trainer.train()

    # Save adapter and tokenizer, plus trainer state and log history
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    try:
        trainer.save_state()
    except Exception as e:
        print(f"[warn] save_state failed: {e}")

    # Write a small run summary
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": os.path.abspath(args.dataset),
        "model_name": args.model_name,
        "output_dir": out_dir,
        "run_name": args.run_name,
        "train_args": training_args,
        "merge_adapter": bool(args.merge_adapter),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        with open(os.path.join(out_dir, "log_history.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    except Exception as e:
        print(f"[warn] failed writing log_history: {e}")

    if args.merge_adapter:
        # Try to merge and save a full model (optional; may require more VRAM)
        try:
            merged_dir = out_dir + "-merged"
            os.makedirs(merged_dir, exist_ok=True)
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
        except Exception as e:
            print(f"[warn] Failed to merge and save full model: {e}")


if __name__ == "__main__":
    main()
    # Gracefully shut down distributed process group to avoid warnings
    try:
        import torch.distributed as dist  # type: ignore
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"[warn] destroy_process_group failed or not applicable: {e}")
