#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq


EXPECTED_FILES = [
    "懂用户.jsonl",
    "懂推荐1.jsonl",
    "懂推荐2.jsonl",
    "懂推荐3.jsonl",
    "懂推荐4.jsonl",
    "懂物料part1.jsonl",
    "懂物料part2.jsonl",
    "懂物料part3.jsonl",
    "懂物料part4.jsonl",
    "懂物料part5.jsonl",
    "懂物料part6.jsonl",
    "懂物料part7.jsonl",
]

SID_RE = re.compile(r"<\|(video|ad|prod|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>")


def task_name(file_name: str) -> str:
    if file_name.startswith("懂用户"):
        return "懂用户"
    if file_name.startswith("懂推荐"):
        return "懂推荐"
    if file_name.startswith("懂物料"):
        return "懂物料"
    return "unknown"


def stable_pick(key: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(16**16 - 1)
    return value < fraction


def resolve_attention_impl(requested: str) -> tuple[str | None, str]:
    if requested == "auto":
        if torch.cuda.is_available():
            return "sdpa", "auto selected PyTorch SDPA for CUDA"
        return None, "auto selected Transformers default for non-CUDA"
    if requested == "default":
        return None, "using Transformers default attention implementation"
    if requested == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError("ATTN_IMPL=flash_attention_2 requires flash-attn; use --attn-impl sdpa if unavailable.")
        return "flash_attention_2", "using flash_attention_2"
    if requested in {"sdpa", "eager"}:
        return requested, f"using {requested}"
    raise ValueError(f"Unsupported attention implementation: {requested}")


def build_sft_features(tokenizer, rec: dict, max_length: int) -> dict:
    messages = []
    if rec.get("system"):
        messages.append({"role": "system", "content": rec["system"]})
    messages.append({"role": "user", "content": rec["prompt"]})
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": rec["response"]}],
        tokenize=False,
        add_generation_prompt=False,
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
        "target_tokens": len(kept_response),
    }


def load_rows(data_dir: Path, eval_fraction: float, seed: int, max_examples: int) -> list[dict]:
    rows = []
    for file_name in EXPECTED_FILES:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run prepare_data.py first or pass --data-dir to prepared JSONL files.")
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                key = f"{file_name}:{line_no}"
                if not stable_pick(key, eval_fraction, seed):
                    continue
                rec = json.loads(line)[0]
                rows.append({
                    "file": file_name,
                    "task": task_name(file_name),
                    "system": rec.get("system", ""),
                    "prompt": rec["prompt"],
                    "response": rec["response"],
                })
                if max_examples and len(rows) >= max_examples:
                    return rows
    return rows


class EvalDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rec = self.rows[index]
        features = build_sft_features(self.tokenizer, rec, self.max_length)
        features.update({"task": rec["task"], "file": rec["file"]})
        return features


def collate_with_meta(tokenizer):
    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    def collate(features):
        meta = {key: [item[key] for item in features] for key in ("target_tokens", "task", "file")}
        tensors = base_collator([{k: v for k, v in item.items() if k not in meta} for item in features])
        tensors.update(meta)
        return tensors

    return collate


def load_model(args, dtype):
    attn_impl, attn_note = resolve_attention_impl(args.attn_impl)
    kwargs = {
        "torch_dtype": dtype if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": args.trust_remote_code,
    }
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    if args.adapter_dir:
        if importlib.util.find_spec("peft") is None:
            raise RuntimeError("--adapter-dir requires peft to be installed.")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()
    return model, {"requested": args.attn_impl, "resolved": attn_impl or "default", "note": attn_note}


def loss_eval(model, dataloader, device):
    totals = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0, "examples": 0})
    with torch.no_grad():
        for batch in dataloader:
            meta = {key: batch.pop(key) for key in ("target_tokens", "task", "file")}
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            labels = batch["labels"]
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            per_token = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            for i, task in enumerate(meta["task"]):
                token_count = int((shift_labels[i] != -100).sum().item())
                if token_count == 0:
                    continue
                loss_sum = float(per_token[i].sum().item())
                for key in ("overall", task, meta["file"][i]):
                    totals[key]["loss_sum"] += loss_sum
                    totals[key]["tokens"] += token_count
                    totals[key]["examples"] += 1
    return {
        key: {
            "examples": value["examples"],
            "tokens": value["tokens"],
            "loss": value["loss_sum"] / max(1, value["tokens"]),
            "ppl": math.exp(min(20.0, value["loss_sum"] / max(1, value["tokens"]))),
        }
        for key, value in sorted(totals.items())
    }


def generation_eval(model, tokenizer, rows: list[dict], args, device):
    results = []
    sample_rows = rows[: args.generation_samples]
    for rec in sample_rows:
        messages = []
        if rec.get("system"):
            messages.append({"role": "system", "content": rec["system"]})
        messages.append({"role": "user", "content": rec["prompt"]})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        expected_has_sid = bool(SID_RE.search(rec["response"]))
        generated_has_sid = bool(SID_RE.search(generated))
        results.append({
            "task": rec["task"],
            "file": rec["file"],
            "expected_has_sid": expected_has_sid,
            "generated_has_sid": generated_has_sid,
            "generated_preview": generated[:500],
        })
    if not results:
        return {"samples": [], "sid_format_rate": None}
    return {
        "samples": results,
        "sid_format_rate": sum(1 for item in results if item["generated_has_sid"]) / len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy local evaluation for LLM-Rec SFT checkpoints; not official benchmark scoring.")
    parser.add_argument("--model-id", default="OpenOneRec/OneReason-0.8B-pretrain-competition")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generation-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-impl", default="auto", choices=["auto", "default", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    if not 0 < args.eval_fraction <= 1:
        raise ValueError("--eval-fraction must be in (0, 1].")

    rows = load_rows(args.data_dir, args.eval_fraction, args.seed, args.max_examples)
    if not rows:
        raise RuntimeError("No eval rows selected; increase --eval-fraction or --max-examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model, attention = load_model(args, dtype)
    device = next(model.parameters()).device

    dataset = EvalDataset(rows, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_meta(tokenizer))
    losses = loss_eval(model, dataloader, device)
    generation = generation_eval(model, tokenizer, rows, args, device) if args.generation_samples else None

    report = {
        "warning": "Proxy local evaluation only. It is not the official OneRec Benchmark score.",
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "data_dir": str(args.data_dir),
        "eval_fraction": args.eval_fraction,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "selected_examples": len(rows),
        "max_length": args.max_length,
        "attention": attention,
        "loss": losses,
        "generation": generation,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
