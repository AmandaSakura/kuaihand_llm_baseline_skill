# kuaihand_llm_baseline_skill

Codex skill for running a local LoRA SFT baseline for the Kuaishou Explorer LLM-Rec / OneReason competition.

This repository is intended for both humans and AI coding agents. It contains:

- `llm-rec-baseline/SKILL.md`: Codex skill instructions.
- `llm-rec-baseline/scripts/run_baseline.sh`: one-command baseline runner.
- `llm-rec-baseline/scripts/prepare_data.py`: unpack and validate the SFT data.
- `llm-rec-baseline/scripts/train_lora_baseline.py`: train a PEFT LoRA adapter.
- `llm-rec-baseline/scripts/local_eval.py`: optional proxy validation loss and generation sanity checks.
- `llm-rec-baseline/scripts/validate_upload.py`: validate Wanqing upload artifacts.
- `llm-rec-baseline/scripts/detect_profile.py`: choose conservative LoRA defaults from hardware.
- `llm-rec-baseline/scripts/check_competition_compliance.py`: reject model configs that do not match the official OneReason-0.8B competition baseline.
- `llm-rec-baseline/scripts/select_sources.py`: probe PyPI/Hugging Face sources and choose reachable endpoints.
- `data/dataset.tar.gz`: official SFT JSONL data (`懂推荐`, `懂物料`, `懂用户`), stored outside the skill folder.

## Install As A Codex Skill

Clone this repository, then copy or symlink only the skill subdirectory into Codex's skill directory:

```bash
mkdir -p ~/.codex/skills
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git ~/kuaihand_llm_baseline_skill
ln -sfn ~/kuaihand_llm_baseline_skill/llm-rec-baseline ~/.codex/skills/llm-rec-baseline
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
SAMPLE_LIMIT=32 MAX_LENGTH=512 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/smoke
```

Run a full baseline:

```bash
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

The script will:

1. Create a Python environment, preferring `uv` if available.
2. Probe package/model sources and choose reachable PyPI/Hugging Face endpoints.
3. Install training dependencies.
4. Reuse prepared data if present, otherwise unpack `data/dataset.tar.gz`.
5. Load `OpenOneRec/OneReason-0.8B-pretrain-competition` from the Hugging Face cache or selected endpoint.
6. Detect hardware and set conservative LoRA defaults.
7. Train a LoRA SFT adapter.
8. Validate upload files and competition config compatibility.

## Environment Manager

Default:

```bash
ENV_MANAGER=auto
```

`auto` uses `uv` if present, otherwise falls back to `python3 -m venv`.

Force `uv`:

```bash
ENV_MANAGER=uv bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

Force stdlib venv:

```bash
ENV_MANAGER=venv bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

## Source Detection And Caching

Default:

```bash
SOURCE_AUTO_DETECT=1
HF_HOME=~/.cache/huggingface
```

Before installing dependencies, the runner probes several PyPI endpoints and selects the first reachable source. It exports both `PIP_INDEX_URL` and `UV_INDEX_URL`, so the chosen source is used by `uv pip` and `pip`.

PyPI candidates:

```text
https://pypi.org/simple
https://pypi.tuna.tsinghua.edu.cn/simple
https://mirrors.ustc.edu.cn/pypi/simple
https://mirrors.aliyun.com/pypi/simple
https://mirrors.cloud.tencent.com/pypi/simple
https://repo.huaweicloud.com/repository/pypi/simple
```

Hugging Face candidates:

```text
https://huggingface.co
https://hf-mirror.com
```

Disable source probing:

```bash
SOURCE_AUTO_DETECT=0 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

Force specific sources:

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
HF_ENDPOINT=https://hf-mirror.com \
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

Repeated runs reuse:

```text
~/.cache/huggingface         # default model cache
<run-dir>/.venv              # existing environment
<run-dir>/data/train_all.jsonl
```

Force recreation:

```bash
RECREATE_ENV=1 FORCE_DATA=1 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
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
PROFILE=gpu8g bash llm-rec-baseline/scripts/run_baseline.sh ./runs/gpu8g
PROFILE=a100-80g bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100
PROFILE=custom MAX_LENGTH=2048 LORA_R=16 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/custom
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
COMPLIANCE_CHECK=1
ATTN_IMPL=auto
```

Use `SAMPLE_LIMIT` only for smoke tests. Keep it `0` for full-data training.

Official offline-training guidance says to use Transformers `v5.3.0`. If that exact package is unavailable in the current Python index, `run_baseline.sh` falls back to `4.53.0`, which matches the released model config.

## Attention Implementation

Default:

```bash
ATTN_IMPL=auto
```

`auto` uses PyTorch `sdpa` on CUDA and the Transformers default on non-CUDA machines. This avoids making `flash-attn` a hard dependency while still avoiding plain eager attention on A100/CUDA runs.

Supported values:

```bash
ATTN_IMPL=auto
ATTN_IMPL=sdpa
ATTN_IMPL=flash_attention_2
ATTN_IMPL=eager
ATTN_IMPL=default
```

Use `flash_attention_2` only if `flash-attn` is already installed and compatible with the current CUDA/PyTorch stack:

```bash
ATTN_IMPL=flash_attention_2 PROFILE=a100-40g \
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-flash
```

Use `eager` only for debugging. Long-context A100 training should use `sdpa` or `flash_attention_2`.

## Competition Compliance

The official guide states that the preliminary round only allows iteration from `OneReason-0.8B`, that evaluation strictly checks the baseline model config, and that contestants may not modify model structure, predefined model parameters, or evaluation settings.

This repository keeps the baseline on the safe path:

- It loads `OpenOneRec/OneReason-0.8B-pretrain-competition` by default.
- It does not add special tokens or resize embeddings.
- It does not change tokenizer, vocab, model config, or architecture.
- It trains a PEFT LoRA adapter by default.
- It writes `adapter_config.json` with the official base model name.
- It validates the base config before training and validates adapter config after training.

Manual check:

```bash
python llm-rec-baseline/scripts/check_competition_compliance.py \
  --model-id OpenOneRec/OneReason-0.8B-pretrain-competition
```

For a local model mirror:

```bash
python llm-rec-baseline/scripts/check_competition_compliance.py \
  --model-id /path/to/OneReason-0.8B-pretrain-competition
```

Do not use this baseline skill for architecture changes such as changing layer count, hidden size, attention heads, vocab size, context config, tokenizer, special tokens, model class, or custom merged model structures unless the competition rules explicitly allow it.

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

## Optional Local Proxy Eval

Official scores are black-box platform results. Local eval in this repository is only a proxy to filter obviously bad runs when you have many candidate adapters.

After training:

```bash
python llm-rec-baseline/scripts/local_eval.py \
  --model-id OpenOneRec/OneReason-0.8B-pretrain-competition \
  --adapter-dir ./runs/baseline/output/lora-baseline \
  --data-dir ./runs/baseline/data \
  --max-length 4096 \
  --max-examples 2048 \
  --generation-samples 16 \
  --output-json ./runs/baseline/local_eval.json
```

It reports:

- overall validation loss and perplexity
- per-task loss for `懂物料`, `懂用户`, `懂推荐`
- per-file loss
- optional greedy generation previews
- semantic-ID token format rate on generated samples

Use this to decide which candidates are worth one of the daily official submissions. Do not treat it as leaderboard score.

## Validate Upload Files Manually

```bash
python llm-rec-baseline/scripts/validate_upload.py \
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
python llm-rec-baseline/scripts/prepare_data.py --output-dir ./runs/data-test
```

This produces:

```text
./runs/data-test/train_all.jsonl
```

## Notes For AI Agents

When using this repository autonomously:

1. Install or reference the `llm-rec-baseline/` subdirectory as the Codex skill; do not install `data/` into the skill folder unless explicitly desired.
2. Prefer `llm-rec-baseline/scripts/run_baseline.sh` for the end-to-end flow.
3. Use `SAMPLE_LIMIT=32` for smoke tests before full training.
4. Do not modify tokenizer, vocab, special tokens, or base model config.
5. Prefer LoRA unless the user explicitly asks for full-parameter training.
6. Keep `COMPLIANCE_CHECK=1` unless the user explicitly asks to experiment outside competition-upload constraints.
7. After training, run `llm-rec-baseline/scripts/validate_upload.py` and report the exact upload files.
