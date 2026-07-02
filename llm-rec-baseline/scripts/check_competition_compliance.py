#!/usr/bin/env python3
import argparse
import json
import os
import urllib.request
from pathlib import Path


OFFICIAL_MODEL_ID = "OpenOneRec/OneReason-0.8B-pretrain-competition"

EXPECTED_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 176253,
    "max_position_embeddings": 40960,
    "tie_word_embeddings": False,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
}


def load_config(model_id: str) -> dict:
    path = Path(model_id)
    if path.exists():
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Local MODEL_ID exists but has no config.json: {config_path}")
        return json.loads(config_path.read_text(encoding="utf-8"))

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{model_id}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_config(config: dict) -> list[str]:
    problems = []
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            problems.append(f"config.{key}: expected {expected!r}, got {actual!r}")
    return problems


def check_adapter_config(adapter_dir: Path) -> list[str]:
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        return [f"Missing adapter_config.json in {adapter_dir}"]
    data = json.loads(adapter_config.read_text(encoding="utf-8"))
    problems = []
    if data.get("task_type") != "CAUSAL_LM":
        problems.append(f"adapter_config.task_type should be 'CAUSAL_LM', got {data.get('task_type')!r}")
    if data.get("peft_type") != "LORA":
        problems.append(f"adapter_config.peft_type should be 'LORA', got {data.get('peft_type')!r}")
    unexpected = set(data.get("modules_to_save") or [])
    if unexpected:
        problems.append(f"adapter_config.modules_to_save should be empty for this baseline, got {sorted(unexpected)!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=OFFICIAL_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = load_config(args.model_id)
    problems = check_config(config)
    if args.adapter_dir:
        problems.extend(check_adapter_config(args.adapter_dir))

    payload = {
        "status": "ok" if not problems else "fail",
        "model_id": args.model_id,
        "official_model_id": OFFICIAL_MODEL_ID,
        "checked_config_keys": sorted(EXPECTED_CONFIG.keys()),
        "problems": problems,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
    else:
        print("[OK] Model config matches the official OneReason-0.8B competition baseline constraints.")

    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
