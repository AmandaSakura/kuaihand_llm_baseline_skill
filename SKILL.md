---
name: llm-rec-baseline
description: Build and run a one-command baseline for the Kuaishou Explorer LLM-Rec / OneReason competition. Use when the user wants to prepare the official OneReason-0.8B competition model, unpack the bundled official SFT JSONL data (懂推荐/懂物料/懂用户), train a LoRA SFT baseline, validate Wanqing upload artifacts, or explain the upload-ready model format.
---

# LLM-Rec Baseline

## Overview

Use this skill to produce a local LoRA SFT baseline for the Kuaishou Explorer LLM-Rec competition from the official OneReason base model and bundled SFT data.

Default to LoRA. It matches Wanqing upload constraints better than full-parameter training and should produce:

```text
adapter_model.safetensors
adapter_config.json
```

## Official Inputs

- Base model: `OpenOneRec/OneReason-0.8B-pretrain-competition`
- Bundled SFT data: `assets/dataset.tar.gz`
- SFT files in the archive: `懂推荐1-4.jsonl`, `懂物料part1-7.jsonl`, `懂用户.jsonl`
- Wanqing upload constraints:
  - LoRA uploads must include `adapter_model.safetensors` and `adapter_config.json`
  - Full-parameter uploads must include `model.safetensors`; if sharded, include `model.safetensors.index.json`
  - Only `.safetensors` and `.json` are accepted
  - Single file must be <= 2 GB
  - Total upload size must be <= 5 GB
  - Maximum upload file count is 20

Do not alter base model architecture, config, tokenizer, vocabulary, or special tokens. The competition evaluation checks compatibility with the baseline model config.

## Workflow

1. Prepare a run directory.
2. Unpack and validate the bundled SFT JSONL data:

```bash
python /Users/joker/.codex/skills/llm-rec-baseline/scripts/prepare_data.py \
  --output-dir /path/to/run/data
```

3. Train the LoRA baseline:

```bash
bash /Users/joker/.codex/skills/llm-rec-baseline/scripts/run_baseline.sh /path/to/run
```

4. Validate upload artifacts:

```bash
python /Users/joker/.codex/skills/llm-rec-baseline/scripts/validate_upload.py \
  --method lora \
  --model-dir /path/to/run/output/lora-baseline
```

## Run Baseline Script

Use `scripts/run_baseline.sh` for the standard end-to-end path. It:

1. Creates or reuses a Python virtual environment under the run directory, preferring `uv` when available.
2. Installs training dependencies with `uv pip` when available, otherwise with `pip`.
3. Unpacks the bundled SFT data.
4. Downloads/loads `OpenOneRec/OneReason-0.8B-pretrain-competition`.
5. Detects available training hardware and fills conservative LoRA defaults.
6. Runs LoRA SFT.
7. Validates the resulting upload files.

Useful environment overrides:

```bash
ENV_MANAGER=auto
PROFILE=auto
MODEL_ID=OpenOneRec/OneReason-0.8B-pretrain-competition
TRANSFORMERS_VERSION=5.3.0
EPOCHS=1
MAX_LENGTH=2048
BATCH_SIZE=1
GRAD_ACCUM=8
LR=2e-4
LORA_R=16
LORA_ALPHA=32
SAMPLE_LIMIT=0
```

Use `SAMPLE_LIMIT` for smoke tests only. Keep it `0` for full-data training.

`ENV_MANAGER=auto` uses `uv` if it is installed and falls back to `python3 -m venv`. Set `ENV_MANAGER=uv` to require uv, or `ENV_MANAGER=venv` to force stdlib venv.

Official offline-training guidance says to use Transformers `v5.3.0`. If that exact package is unavailable in the current Python index, `run_baseline.sh` falls back to `4.53.0`, which matches the released model config, and prints a warning.

`PROFILE=auto` detects CUDA, Apple MPS, or CPU and sets defaults only for unset variables. Use `PROFILE=custom` to disable hardware defaults entirely. Common explicit profiles are:

- `gpu8g`: CUDA GPU with about 8 GB VRAM, e.g. RTX 4060. Defaults to `MAX_LENGTH=1024`, `BATCH_SIZE=1`, `GRAD_ACCUM=8`, `LORA_R=16`.
- `mps16g`: Apple Silicon with unified memory around 16 GB. Defaults to `MAX_LENGTH=512`, `BATCH_SIZE=1`, `GRAD_ACCUM=8`, `LORA_R=8`.
- `a100-40g`: single A100 40 GB. Defaults to `MAX_LENGTH=4096`, `BATCH_SIZE=2`, `GRAD_ACCUM=4`, `LORA_R=32`.
- `a100-80g`: single A100 80 GB. Defaults to `MAX_LENGTH=8192`, `BATCH_SIZE=2`, `GRAD_ACCUM=4`, `LORA_R=32`.
- `cpu`: CPU-only fallback. Defaults to `MAX_LENGTH=384`, `BATCH_SIZE=1`, `GRAD_ACCUM=16`, `LORA_R=4`.

For a smoke test, set `SAMPLE_LIMIT=32` manually; auto profiles do not silently reduce the training set.

## Upload Guidance

For Wanqing, upload only the validated LoRA files from the output directory:

```text
adapter_model.safetensors
adapter_config.json
```

In the Wanqing UI, choose the training method as LoRA. Use a short model name and version such as `llm-rec-lora-baseline` / `V1`.

Do not upload optimizer checkpoints, trainer state, tokenizer files, logs, or intermediate checkpoints unless Wanqing explicitly changes the requirements.
