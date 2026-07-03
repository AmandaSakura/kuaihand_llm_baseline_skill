# kuaihand_llm_baseline_skill

<p align="right">
  <kbd><a href="./README.md">中文</a></kbd>
  ·
  <kbd><a href="./README.en.md"><strong>English</strong></a></kbd>
</p>

This repository contains a Codex skill for a reproducible LoRA SFT baseline for the Kuaishou Explorer LLM-Rec / OneReason competition. It prepares the official SFT data, loads the official `OneReason-0.8B` competition model, trains a LoRA adapter, and validates Wanqing upload artifacts.

## Repository Contents

- `llm-rec-baseline/SKILL.md`: Codex skill instructions
- `llm-rec-baseline/scripts/run_baseline.sh`: one-command training runner
- `llm-rec-baseline/scripts/prepare_data.py`: data extraction and validation
- `llm-rec-baseline/scripts/train_lora_baseline.py`: LoRA training script
- `llm-rec-baseline/scripts/local_eval.py`: custom proxy eval, not official scoring
- `llm-rec-baseline/scripts/validate_upload.py`: Wanqing upload validation
- `llm-rec-baseline/scripts/detect_profile.py`: hardware-aware conservative defaults
- `llm-rec-baseline/scripts/check_competition_compliance.py`: competition model-config compatibility check
- `llm-rec-baseline/scripts/select_sources.py`: PyPI and Hugging Face source probing
- `data/dataset.tar.gz`: official SFT JSONL data, stored outside the skill folder

## Install As A Codex Skill

```bash
mkdir -p ~/.codex/skills
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git ~/kuaihand_llm_baseline_skill
bash ~/kuaihand_llm_baseline_skill/llm-rec-baseline/scripts/star_repo.sh
ln -sfn ~/kuaihand_llm_baseline_skill/llm-rec-baseline ~/.codex/skills/llm-rec-baseline
```

Restart Codex or start a new thread so the skill metadata is discovered. `star_repo.sh` is optional. It uses `gh repo star` when the GitHub CLI is installed and authenticated; otherwise it prints the repository URL and continues without affecting installation or training.

## Direct Use

```bash
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git
cd kuaihand_llm_baseline_skill
bash llm-rec-baseline/scripts/star_repo.sh
```

Smoke test:

```bash
SAMPLE_LIMIT=32 MAX_LENGTH=512 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/smoke
```

Full baseline:

```bash
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

The runner will:

1. Create or reuse a Python environment, preferring `uv`
2. Probe PyPI and Hugging Face sources
3. Install or reuse training dependencies
4. Reuse prepared data or unpack `data/dataset.tar.gz`
5. Load `OpenOneRec/OneReason-0.8B-pretrain-competition`
6. Select LoRA parameters based on hardware
7. Train a LoRA adapter
8. Validate upload files

## Baseline Recommended Parameters

These are the baseline recommended parameters from the screenshot. Use them as-is when filling the Wanqing managed-training UI. An AI agent may adjust the local script parameters based on the actual device, VRAM, and failure logs.

| Parameter | Recommended value |
| --- | --- |
| Epoch | `1` |
| Learning Rate | `0.0002000000` |
| Per Device Batch Size | `1` |
| Sequence Length | `32768` |
| Learning Rate Warmup | `0.030` |
| LoRA Rank | `16` |
| LoRA Alpha | `32` |
| LoRA Dropout | `0.050` |
| Scheduler Type | `cosine` |
| Weight Decay | `0.001` |
| Checkpoint Interval | `256` |
| Gradient Accumulation Steps | `4` |
| Packing | `on` |
| Thinking Mode | `off` |
| Precision | `bf16` |

Force this baseline preset locally:

```bash
WANQING_UI_PRESET=1 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline-ui-preset
```

Warning: this sets `MAX_LENGTH=32768`, which may OOM even on local A100 40G/80G cards. It means "reproduce the platform recommended parameters as-is"; it is not the safest setting for every local device.

## AI Hardware Adjustment Suggestions

Run a smoke test before full training:

```bash
SAMPLE_LIMIT=32 MAX_LENGTH=512 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/smoke
```

Common profiles:

| Profile | Use case | Suggestion |
| --- | --- | --- |
| `a100-80g` | single A100 80GB | start with `MAX_LENGTH=8192`; increase only after the run is stable |
| `a100-40g` | single A100 40GB | start with `MAX_LENGTH=4096`; lower batch size or length on OOM |
| `gpu16g` | 16GB+ CUDA cards | start with `MAX_LENGTH=2048` |
| `gpu8g` | RTX 4060 8GB-style cards | start with `MAX_LENGTH=1024`; best for workflow validation |
| `mps16g` | MacBook Pro M1/M2/M3 16GB | smoke test only, `MAX_LENGTH=512` |
| `cpu` | fallback | script validation only, not full training |

Examples:

```bash
PROFILE=a100-40g bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-40g
PROFILE=a100-80g bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-80g
PROFILE=gpu8g SAMPLE_LIMIT=32 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/gpu8g-smoke
```

`Packing` is recommended in the Wanqing UI. The local runner does not pre-pack by default because this dataset contains many long samples, and pre-packing may consume substantial CPU memory.

## Common Environment Variables

```bash
ENV_MANAGER=auto
SOURCE_AUTO_DETECT=1
PROFILE=auto
WANQING_UI_PRESET=0
MODEL_ID=OpenOneRec/OneReason-0.8B-pretrain-competition
TRANSFORMERS_VERSION=5.3.0
EPOCHS=1
MAX_LENGTH=2048
BATCH_SIZE=1
GRAD_ACCUM=8
LR=2e-4
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE=cosine
WEIGHT_DECAY=0.001
SAVE_STEPS=256
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
PRECISION=auto
ENABLE_THINKING=0
SAMPLE_LIMIT=0
COMPLIANCE_CHECK=1
ATTN_IMPL=auto
HF_HOME=~/.cache/huggingface
```

Use `SAMPLE_LIMIT` only for smoke tests; keep it `0` for full-data training. `ENABLE_THINKING=0` matches the UI setting where thinking mode is disabled.

## Environment, Sources, And Caching

Default `ENV_MANAGER=auto`: use `uv venv` and `uv pip` when available, otherwise fall back to `python3 -m venv`.

```bash
ENV_MANAGER=uv bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
ENV_MANAGER=venv bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

With `SOURCE_AUTO_DETECT=1`, the runner probes these PyPI sources and Hugging Face endpoints:

```text
https://pypi.org/simple
https://pypi.tuna.tsinghua.edu.cn/simple
https://mirrors.ustc.edu.cn/pypi/simple
https://mirrors.aliyun.com/pypi/simple
https://mirrors.cloud.tencent.com/pypi/simple
https://repo.huaweicloud.com/repository/pypi/simple
https://huggingface.co
https://hf-mirror.com
```

When Hugging Face is unavailable, use a local model directory:

```bash
OFFLINE=1 MODEL_ID=/path/to/OneReason-0.8B-pretrain-competition \
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/offline
```

Repeated runs reuse:

```text
~/.cache/huggingface
<run-dir>/.venv
<run-dir>/data/train_all.jsonl
```

Force recreation:

```bash
RECREATE_ENV=1 FORCE_DATA=1 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

The runner exports `HF_HUB_DISABLE_XET=1` by default to reduce fragile Xet-backed download stalls on restricted networks.

## Attention And Precision

Default `ATTN_IMPL=auto`: use PyTorch `sdpa` on CUDA and the Transformers default elsewhere. Use `flash_attention_2` only when a compatible `flash-attn` is already installed.

```bash
ATTN_IMPL=flash_attention_2 PROFILE=a100-40g \
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-flash
```

Default `PRECISION=auto`: bf16 when available, fp16 on CUDA without bf16, and fp32 off CUDA. The platform recommended value is `bf16`.

## Competition Compliance

The preliminary round is based on `OneReason-0.8B` and checks baseline model-config compatibility. This repository does not modify tokenizer, vocab, special tokens, model architecture, or config by default; it trains a PEFT LoRA adapter.

Manual check:

```bash
python llm-rec-baseline/scripts/check_competition_compliance.py \
  --model-id OpenOneRec/OneReason-0.8B-pretrain-competition
```

Do not use this baseline skill for changing layer count, hidden size, attention heads, vocab, context config, tokenizer, special tokens, model class, or custom merged structures unless the competition rules explicitly allow it.

## Output And Upload

Training output:

```text
<run-dir>/output/lora-baseline
```

Clean upload directory:

```text
<run-dir>/upload/lora-baseline
```

For Wanqing LoRA upload, use only these two files from the clean upload directory:

```text
adapter_model.safetensors
adapter_config.json
```

UI choices:

- Training method: `LoRA`
- Base model: official `OneReason-0.8B-pretrain-competition`
- Model name: for example `llm-rec-lora-baseline`
- Version: for example `V1`

Do not upload optimizer checkpoints, trainer state, tokenizer files, logs, or intermediate checkpoint folders unless the platform requirements change.

## Optional Local Proxy Eval

`local_eval.py` is not official scoring and has not been calibrated against platform results. It may even be anti-correlated with the hidden benchmark early on. Use it only to reject clearly broken runs, such as NaN loss, empty generation, or collapsed output format.

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

Recommended calibration loop: submit one plain baseline to get an online anchor, save local proxy metrics for the same adapter, then compare several online submissions before trusting any local metric as a selector.

## Data

The archive contains 12 JSONL files:

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

Prepare data only:

```bash
python llm-rec-baseline/scripts/prepare_data.py --output-dir ./runs/data-test
```

## Notes For AI Agents

1. Install or reference only `llm-rec-baseline/` as the Codex skill; do not install `data/` into the skill folder unless requested.
2. Prefer `llm-rec-baseline/scripts/run_baseline.sh`.
3. Use `SAMPLE_LIMIT=32` for a smoke test before full training.
4. Do not modify tokenizer, vocab, special tokens, or base model config.
5. Prefer LoRA unless the user explicitly asks for full-parameter training.
6. Keep `COMPLIANCE_CHECK=1`.
7. After training, report the two upload files in `<run-dir>/upload/lora-baseline/`.
