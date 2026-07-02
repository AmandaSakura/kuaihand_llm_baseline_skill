#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


MAX_SINGLE = 2 * 1000**3
MAX_TOTAL = 5 * 1000**3
MAX_FILES = 20


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def check_file(path: Path) -> int:
    if not path.exists():
        fail(f"Missing required file: {path}")
    if path.suffix not in (".safetensors", ".json"):
        fail(f"Unsupported upload suffix: {path.name}")
    size = path.stat().st_size
    if size > MAX_SINGLE:
        fail(f"{path.name} is larger than 2 GB")
    return size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "full"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.method == "lora":
        files = [
            args.model_dir / "adapter_model.safetensors",
            args.model_dir / "adapter_config.json",
        ]
    else:
        single = args.model_dir / "model.safetensors"
        index = args.model_dir / "model.safetensors.index.json"
        if single.exists():
            files = [single]
        elif index.exists():
            with index.open("r", encoding="utf-8") as fh:
                idx = json.load(fh)
            files = [index] + sorted({args.model_dir / name for name in idx.get("weight_map", {}).values()})
        else:
            fail("Full-parameter upload requires model.safetensors or model.safetensors.index.json")

    if len(files) > MAX_FILES:
        fail(f"Too many upload files: {len(files)} > {MAX_FILES}")

    total = sum(check_file(path) for path in files)
    if total > MAX_TOTAL:
        fail(f"Upload total is larger than 5 GB: {total}")

    print(json.dumps({
        "status": "ok",
        "method": args.method,
        "model_dir": str(args.model_dir),
        "file_count": len(files),
        "total_bytes": total,
        "upload_files": [path.name for path in files],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
