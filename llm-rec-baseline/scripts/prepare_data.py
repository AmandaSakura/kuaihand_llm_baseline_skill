#!/usr/bin/env python3
import argparse
import gzip
import json
import shutil
import tarfile
from pathlib import Path


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


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_archive() -> Path:
    candidates = []
    if "DATASET_ARCHIVE" in __import__("os").environ:
        candidates.append(Path(__import__("os").environ["DATASET_ARCHIVE"]))
    root = skill_root()
    candidates.extend([
        root / "assets" / "dataset.tar.gz",
        root.parent / "data" / "dataset.tar.gz",
        root.parent / "assets" / "dataset.tar.gz",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return root / "assets" / "dataset.tar.gz"


def safe_extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = output_dir / member.name
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(output_dir)


def validate_record(record, file_name: str, line_no: int) -> None:
    if not isinstance(record, list) or len(record) != 1:
        raise ValueError(f"{file_name}:{line_no} must be a one-element JSON array")
    obj = record[0]
    if not isinstance(obj, dict):
        raise ValueError(f"{file_name}:{line_no} array element must be an object")
    for key in ("system", "prompt", "response"):
        if key not in obj:
            raise ValueError(f"{file_name}:{line_no} missing {key!r}")
        if not isinstance(obj[key], str):
            raise ValueError(f"{file_name}:{line_no} {key!r} must be a string")


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def validate_dataset(data_dir: Path, sample_lines: int) -> dict:
    stats = {}
    for name in EXPECTED_FILES:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")
        count = 0
        with open_text(path) as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                count += 1
                if line_no <= sample_lines:
                    validate_record(json.loads(line), name, line_no)
        stats[name] = count
    return stats


def combine_dataset(data_dir: Path, combined_path: Path) -> int:
    total = 0
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as out:
        for name in EXPECTED_FILES:
            with (data_dir / name).open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line.rstrip("\n") + "\n")
                        total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=default_archive())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-name", default="train_all.jsonl")
    parser.add_argument("--sample-lines", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    combined = args.output_dir / args.combined_name
    if not args.force and combined.exists():
        try:
            stats = validate_dataset(args.output_dir, args.sample_lines)
            total = sum(stats.values())
            print(json.dumps({
                "data_dir": str(args.output_dir),
                "combined": str(combined),
                "files": stats,
                "total_rows": total,
                "reused": True,
            }, ensure_ascii=False, indent=2))
            return
        except Exception:
            pass

    if not args.archive.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {args.archive}. Set DATASET_ARCHIVE=/path/to/dataset.tar.gz "
            "or place it at ../data/dataset.tar.gz relative to the skill folder."
        )
    if args.force and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    safe_extract(args.archive, args.output_dir)
    stats = validate_dataset(args.output_dir, args.sample_lines)
    total = combine_dataset(args.output_dir, combined)

    print(json.dumps({
        "data_dir": str(args.output_dir),
        "combined": str(combined),
        "files": stats,
        "total_rows": total,
        "reused": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
