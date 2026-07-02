#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-$PWD/llm-rec-baseline-run}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$RUN_DIR/data"
OUTPUT_DIR="$RUN_DIR/output/lora-baseline"
VENV_DIR="$RUN_DIR/.venv"

MODEL_ID="${MODEL_ID:-OpenOneRec/OneReason-0.8B-pretrain-competition}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.3.0}"
PROFILE="${PROFILE:-auto}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
SOURCE_AUTO_DETECT="${SOURCE_AUTO_DETECT:-1}"
RECREATE_ENV="${RECREATE_ENV:-0}"
FORCE_DATA="${FORCE_DATA:-0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HOME

mkdir -p "$RUN_DIR"

if [[ "$SOURCE_AUTO_DETECT" == "1" ]]; then
  eval "$(python3 "$SKILL_DIR/scripts/select_sources.py" --format shell)"
fi

if [[ -x "$VENV_DIR/bin/python" && "$RECREATE_ENV" != "1" ]]; then
  ENV_MANAGER_RESOLVED="existing"
  PIP_INSTALL=("$VENV_DIR/bin/python" -m pip install)
elif [[ "$ENV_MANAGER" == "uv" ]] || { [[ "$ENV_MANAGER" == "auto" ]] && command -v uv >/dev/null 2>&1; }; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: ENV_MANAGER=uv was requested, but uv was not found on PATH." >&2
    exit 1
  fi
  uv venv "$VENV_DIR"
  PIP_INSTALL=(uv pip install --python "$VENV_DIR/bin/python")
  ENV_MANAGER_RESOLVED="uv"
elif [[ "$ENV_MANAGER" == "venv" ]] || [[ "$ENV_MANAGER" == "auto" ]]; then
  python3 -m venv "$VENV_DIR"
  PIP_INSTALL=("$VENV_DIR/bin/python" -m pip install)
  ENV_MANAGER_RESOLVED="venv"
else
  echo "ERROR: ENV_MANAGER must be auto, uv, or venv; got $ENV_MANAGER" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

"${PIP_INSTALL[@]}" --upgrade pip
"${PIP_INSTALL[@]}" "torch" "accelerate" "peft" "safetensors" "datasets"
if ! "${PIP_INSTALL[@]}" "transformers==$TRANSFORMERS_VERSION"; then
  if [[ "$TRANSFORMERS_VERSION" == "5.3.0" ]]; then
    echo "WARNING: transformers==5.3.0 is unavailable; falling back to transformers==4.53.0 from the released model config." >&2
    "${PIP_INSTALL[@]}" "transformers==4.53.0"
  else
    exit 1
  fi
fi

if [[ "$PROFILE" != "custom" ]]; then
  eval "$(python "$SKILL_DIR/scripts/detect_profile.py" --profile "$PROFILE" --format shell)"
fi

EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
COMPLIANCE_CHECK="${COMPLIANCE_CHECK:-1}"
ATTN_IMPL="${ATTN_IMPL:-auto}"

echo "Training profile: ${RESOLVED_PROFILE:-custom}"
echo "Environment manager: $ENV_MANAGER_RESOLVED ($VENV_DIR)"
echo "Package source: ${SELECTED_PYPI_SOURCE:-existing-env-or-default} ${PIP_INDEX_URL:-}"
echo "Hugging Face source: ${SELECTED_HF_SOURCE:-default/cache} ${HF_ENDPOINT:-https://huggingface.co}"
echo "HF cache: $HF_HOME"
echo "LoRA params: MAX_LENGTH=$MAX_LENGTH BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM LR=$LR LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA SAMPLE_LIMIT=$SAMPLE_LIMIT ATTN_IMPL=$ATTN_IMPL"

PREPARE_ARGS=(python "$SKILL_DIR/scripts/prepare_data.py" --output-dir "$DATA_DIR")
if [[ "$FORCE_DATA" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
"${PREPARE_ARGS[@]}"

if [[ "$COMPLIANCE_CHECK" == "1" ]]; then
  python "$SKILL_DIR/scripts/check_competition_compliance.py" --model-id "$MODEL_ID"
fi

python "$SKILL_DIR/scripts/train_lora_baseline.py" \
  --model-id "$MODEL_ID" \
  --train-file "$DATA_DIR/train_all.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --max-length "$MAX_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --adapter-base-model-name "OpenOneRec/OneReason-0.8B-pretrain-competition" \
  --attn-impl "$ATTN_IMPL" \
  --sample-limit "$SAMPLE_LIMIT"

python "$SKILL_DIR/scripts/validate_upload.py" \
  --method lora \
  --model-dir "$OUTPUT_DIR"

if [[ "$COMPLIANCE_CHECK" == "1" ]]; then
  python "$SKILL_DIR/scripts/check_competition_compliance.py" \
    --model-id "$MODEL_ID" \
    --adapter-dir "$OUTPUT_DIR"
fi

echo "Upload-ready LoRA files are in: $OUTPUT_DIR"
