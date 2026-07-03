#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def apply_chat_template(tokenizer, messages: list[dict], *, add_generation_prompt: bool, enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def resolve_attention_impl(requested: str) -> tuple[str | None, str]:
    if requested == "auto":
        if torch.cuda.is_available():
            return "sdpa", "auto selected PyTorch SDPA for CUDA"
        return None, "auto selected Transformers default for non-CUDA"
    if requested == "default":
        return None, "using Transformers default attention implementation"
    if requested == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "ATTN_IMPL=flash_attention_2 requires the flash-attn package. "
                "Install it in the run environment or use ATTN_IMPL=sdpa."
            )
        return "flash_attention_2", "using flash_attention_2"
    if requested in {"sdpa", "eager"}:
        return requested, f"using {requested}"
    raise ValueError(f"Unsupported attention implementation: {requested}")


def resolve_precision(requested: str) -> tuple[torch.dtype, bool, bool, str]:
    cuda = torch.cuda.is_available()
    bf16_supported = cuda and torch.cuda.is_bf16_supported()
    if requested == "auto":
        if bf16_supported:
            return torch.bfloat16, True, False, "auto selected bf16"
        if cuda:
            return torch.float16, False, True, "auto selected fp16"
        return torch.float32, False, False, "auto selected fp32"
    if requested == "bf16":
        if not bf16_supported:
            raise RuntimeError("PRECISION=bf16 requires a CUDA GPU with bf16 support.")
        return torch.bfloat16, True, False, "using bf16"
    if requested == "fp16":
        if not cuda:
            raise RuntimeError("PRECISION=fp16 requires CUDA in this baseline script.")
        return torch.float16, False, True, "using fp16"
    if requested == "fp32":
        return torch.float32, False, False, "using fp32"
    raise ValueError(f"Unsupported precision: {requested}")


def build_sft_features(tokenizer, rec: dict, max_length: int, enable_thinking: bool = False) -> dict:
    messages = []
    if rec.get("system"):
        messages.append({"role": "system", "content": rec["system"]})
    messages.append({"role": "user", "content": rec["prompt"]})

    prompt_text = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = apply_chat_template(
        tokenizer,
        messages + [{"role": "assistant", "content": rec["response"]}],
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    response_ids = full_ids[len(prompt_ids):]
    if not response_ids:
        response_ids = tokenizer(rec["response"], add_special_tokens=False)["input_ids"]

    if len(prompt_ids) + len(response_ids) <= max_length:
        kept_prompt = prompt_ids
        kept_response = response_ids
    else:
        kept_response = response_ids[:max_length]
        prompt_budget = max(0, max_length - len(kept_response))
        kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget else []

    input_ids = kept_prompt + kept_response
    labels = [-100] * len(kept_prompt) + kept_response
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class SftJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, sample_limit: int = 0, enable_thinking: bool = False):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)[0]
                self.rows.append(rec)
                if sample_limit and len(self.rows) >= sample_limit:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return build_sft_features(self.tokenizer, self.rows[index], self.max_length, self.enable_thinking)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="OpenOneRec/OneReason-0.8B-pretrain-competition")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--adapter-base-model-name", default="OpenOneRec/OneReason-0.8B-pretrain-competition")
    parser.add_argument("--attn-impl", default="auto", choices=["auto", "default", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype, use_bf16, use_fp16, precision_note = resolve_precision(args.precision)
    attn_impl, attn_note = resolve_attention_impl(args.attn_impl)
    model_kwargs = {
        "torch_dtype": dtype if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": args.trust_remote_code,
    }
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    print(json.dumps({
        "attention": {
            "requested": args.attn_impl,
            "resolved": attn_impl or "default",
            "note": attn_note,
            "cuda": torch.cuda.is_available(),
        },
        "precision": {
            "requested": args.precision,
            "resolved": str(dtype).replace("torch.", ""),
            "note": precision_note,
        },
        "enable_thinking": args.enable_thinking,
    }, ensure_ascii=False))
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SftJsonlDataset(args.train_file, tokenizer, args.max_length, args.sample_limit, args.enable_thinking)
    if len(dataset) == 0:
        raise RuntimeError(f"No training rows loaded from {args.train_file}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    adapter_config_path = args.output_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = args.adapter_base_model_name
        adapter_config_path.write_text(json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "upload_files": ["adapter_model.safetensors", "adapter_config.json"],
        "rows": len(dataset),
        "attn_implementation": attn_impl or "default",
        "precision": str(dtype).replace("torch.", ""),
        "enable_thinking": args.enable_thinking,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
