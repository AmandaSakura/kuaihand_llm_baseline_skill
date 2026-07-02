# kuaihand_llm_baseline_skill

Codex skill for running a local LoRA SFT baseline for the Kuaishou Explorer LLM-Rec / OneReason competition.

This repository is intended for both humans and AI coding agents. It contains:

- `SKILL.md`: Codex skill instructions.
- `assets/dataset.tar.gz`: bundled official SFT JSONL data (`懂推荐`, `懂物料`, `懂用户`).
- `scripts/run_baseline.sh`: one-command baseline runner.
- `scripts/prepare_data.py`: unpack and validate the bundled data.
- `scripts/train_lora_baseline.py`: train a PEFT LoRA adapter.
- `scripts/validate_upload.py`: validate Wanqing upload artifacts.
- `scripts/detect_profile.py`: choose conservative LoRA defaults from hardware.

## Install As A Codex Skill

Clone this repository into Codex's skill directory:

```bash
mkdir -p ~/.codex/skills
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git ~/.codex/skills/llm-rec-baseline
```

Restart Codex or start a new thread so the skill metadata is discovered.

After installation, ask Codex something like:

```text
Use the llm-rec-baseline skill to run a smoke test.
```

or:

```text
Use the llm-rec-baseline skill to train the OneReason LoRA baseline on this machine.
```

## Direct Use Without Codex

Clone anywhere:

```bash
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git
cd kuaihand_llm_baseline_skill
```

Run a quick smoke test:

```bash
SAMPLE_LIMIT=32 MAX_LENGTH=512 bash scripts/run_baseline.sh ./runs/smoke
```

Run a full baseline:

```bash
bash scripts/run_baseline.sh ./runs/baseline
```

The script will:

1. Create a Python environment, preferring `uv` if available.
2. Install training dependencies.
3. Unpack `assets/dataset.tar.gz`.
4. Load `OpenOneRec/OneReason-0.8B-pretrain-competition`.
5. Detect hardware and set conservative LoRA defaults.
6. Train a LoRA SFT adapter.
7. Validate the upload files.

## Environment Manager

Default:

```bash
ENV_MANAGER=auto
```

`auto` uses `uv` if present, otherwise falls back to `python3 -m venv`.

Force `uv`:

```bash
ENV_MANAGER=uv bash scripts/run_baseline.sh ./runs/baseline
```

Force stdlib venv:

```bash
ENV_MANAGER=venv bash scripts/run_baseline.sh ./runs/baseline
```

## Hardware Profiles

Default:

```bash
PROFILE=auto
```

`auto` detects CUDA, Apple MPS, or CPU and sets defaults only for variables that are not already set.

Explicit profiles:

```bash
PROFILE=gpu8g      # RTX 4060 8GB style cards
PROFILE=gpu16g     # generic 16GB+ CUDA cards
PROFILE=a100-40g   # single A100 40GB
PROFILE=a100-80g   # single A100 80GB
PROFILE=mps16g     # Apple Silicon 16GB unified memory
PROFILE=cpu        # CPU fallback
PROFILE=custom     # disable automatic defaults
```

Examples:

```bash
PROFILE=gpu8g bash scripts/run_baseline.sh ./runs/gpu8g
PROFILE=a100-80g bash scripts/run_baseline.sh ./runs/a100
PROFILE=custom MAX_LENGTH=2048 LORA_R=16 bash scripts/run_baseline.sh ./runs/custom
```

## Common Training Overrides

```bash
MODEL_ID=OpenOneRec/OneReason-0.8B-pretrain-competition
TRANSFORMERS_VERSION=5.3.0
EPOCHS=1
MAX_LENGTH=1024
BATCH_SIZE=1
GRAD_ACCUM=8
LR=2e-4
LORA_R=16
LORA_ALPHA=32
SAMPLE_LIMIT=0
```

Use `SAMPLE_LIMIT` only for smoke tests. Keep it `0` for full-data training.

Official offline-training guidance says to use Transformers `v5.3.0`. If that exact package is unavailable in the current Python index, `run_baseline.sh` falls back to `4.53.0`, which matches the released model config.

## Output And Upload

The default output directory is:

```text
<run-dir>/output/lora-baseline
```

For Wanqing LoRA upload, use only:

```text
adapter_model.safetensors
adapter_config.json
```

In the Wanqing upload UI:

- Training method: `LoRA`
- Model source/base: official `OneReason-0.8B-pretrain-competition`
- Model name: for example `llm-rec-lora-baseline`
- Version: for example `V1`

Do not upload optimizer checkpoints, trainer state, tokenizer files, logs, or intermediate checkpoint folders unless the platform requirements change.

## Validate Upload Files Manually

```bash
python scripts/validate_upload.py \
  --method lora \
  --model-dir ./runs/baseline/output/lora-baseline
```

The validator checks:

- required LoRA files are present
- only `.safetensors` and `.json` files are used
- each file is <= 2 GB
- total upload size is <= 5 GB
- file count is <= 20

## Data

The bundled archive contains 12 JSONL files:

```text
懂用户.jsonl
懂推荐1.jsonl
懂推荐2.jsonl
懂推荐3.jsonl
懂推荐4.jsonl
懂物料part1.jsonl
懂物料part2.jsonl
懂物料part3.jsonl
懂物料part4.jsonl
懂物料part5.jsonl
懂物料part6.jsonl
懂物料part7.jsonl
```

Prepare only the data:

```bash
python scripts/prepare_data.py --output-dir ./runs/data-test
```

This produces:

```text
./runs/data-test/train_all.jsonl
```

## Notes For AI Agents

When using this repository autonomously:

1. Prefer `scripts/run_baseline.sh` for the end-to-end flow.
2. Use `SAMPLE_LIMIT=32` for smoke tests before full training.
3. Do not modify tokenizer, vocab, special tokens, or base model config.
4. Prefer LoRA unless the user explicitly asks for full-parameter training.
5. After training, run `scripts/validate_upload.py` and report the exact upload files.
