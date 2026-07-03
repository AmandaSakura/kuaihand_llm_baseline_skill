#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-$PWD/llm-rec-baseline-run}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$RUN_DIR/data"
OUTPUT_DIR="$RUN_DIR/output/lora-baseline"
UPLOAD_DIR="$RUN_DIR/upload/lora-baseline"
VENV_DIR="$RUN_DIR/.venv"

MODEL_ID="${MODEL_ID:-OpenOneRec/OneReason-0.8B-pretrain-competition}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.3.0}"
PROFILE="${PROFILE:-auto}"
WANQING_UI_PRESET="${WANQING_UI_PRESET:-0}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
SOURCE_AUTO_DETECT="${SOURCE_AUTO_DETECT:-1}"
RECREATE_ENV="${RECREATE_ENV:-0}"
FORCE_DATA="${FORCE_DATA:-0}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
OFFLINE="${OFFLINE:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-auto}"
export HF_HOME HF_HUB_DISABLE_XET

mkdir -p "$RUN_DIR"

if [[ "$OFFLINE" == "1" ]]; then
  if [[ ! -d "$MODEL_ID" ]]; then
    echo "ERROR: OFFLINE=1 requires MODEL_ID to be a local model directory; got $MODEL_ID" >&2
    exit 1
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  SOURCE_AUTO_DETECT=0
fi

if [[ "$SOURCE_AUTO_DETECT" == "1" ]]; then
  eval "$(python3 "$SKILL_DIR/scripts/select_sources.py" --format shell)"
fi

if [[ "$RECREATE_ENV" == "1" ]]; then
  rm -rf "$VENV_DIR"
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

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 is required, got {sys.version.split()[0]}")
PY

DEPS_READY=0
if [[ "$INSTALL_DEPS" == "auto" || "$INSTALL_DEPS" == "0" ]]; then
  if python - "$TRANSFORMERS_VERSION" <<'PY'
import importlib.util
import sys
required = ["torch", "accelerate", "peft", "safetensors", "datasets", "jinja2", "transformers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("missing deps:", ",".join(missing))
    raise SystemExit(1)
import transformers
want = sys.argv[1]
allowed = {want}
if want == "5.3.0":
    allowed.add("4.53.0")
if transformers.__version__ not in allowed:
    print(f"transformers version mismatch: have {transformers.__version__}, want {want}")
    raise SystemExit(1)
print("dependencies already available")
PY
  then
    DEPS_READY=1
  fi
fi

if [[ "$INSTALL_DEPS" == "0" && "$DEPS_READY" != "1" ]]; then
  echo "ERROR: INSTALL_DEPS=0 but required Python packages are missing or incompatible." >&2
  exit 1
fi

if [[ "$DEPS_READY" != "1" ]]; then
  "${PIP_INSTALL[@]}" --upgrade pip
  "${PIP_INSTALL[@]}" "torch" "accelerate" "peft" "safetensors" "datasets" "jinja2"
  if ! "${PIP_INSTALL[@]}" "transformers==$TRANSFORMERS_VERSION"; then
    if [[ "$TRANSFORMERS_VERSION" == "5.3.0" ]]; then
      echo "WARNING: transformers==5.3.0 is unavailable; falling back to transformers==4.53.0 from the released model config." >&2
      "${PIP_INSTALL[@]}" "transformers==4.53.0"
    else
      exit 1
    fi
  fi
fi

if [[ "$PROFILE" != "custom" ]]; then
  eval "$(python "$SKILL_DIR/scripts/detect_profile.py" --profile "$PROFILE" --format shell)"
fi

if [[ "$WANQING_UI_PRESET" == "1" ]]; then
  EPOCHS="1"
  LR="2e-4"
  BATCH_SIZE="1"
  MAX_LENGTH="32768"
  WARMUP_RATIO="0.03"
  LORA_R="16"
  LORA_ALPHA="32"
  LORA_DROPOUT="0.05"
  LR_SCHEDULER_TYPE="cosine"
  WEIGHT_DECAY="0.001"
  SAVE_STEPS="256"
  GRAD_ACCUM="4"
  PRECISION="bf16"
  ENABLE_THINKING="0"
  echo "WARNING: WANQING_UI_PRESET=1 matches the managed-platform UI recommendation and may OOM on local GPUs." >&2
fi

EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
SAVE_STEPS="${SAVE_STEPS:-256}"
PRECISION="${PRECISION:-auto}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
COMPLIANCE_CHECK="${COMPLIANCE_CHECK:-1}"
ATTN_IMPL="${ATTN_IMPL:-auto}"

echo "Training profile: ${RESOLVED_PROFILE:-custom}"
echo "Environment manager: $ENV_MANAGER_RESOLVED ($VENV_DIR)"
echo "Package source: ${SELECTED_PYPI_SOURCE:-existing-env-or-default} ${PIP_INDEX_URL:-}"
echo "Hugging Face source: ${SELECTED_HF_SOURCE:-default/cache} ${HF_ENDPOINT:-https://huggingface.co}"
echo "HF cache: $HF_HOME"
echo "HF xet disabled: $HF_HUB_DISABLE_XET; offline: $OFFLINE"
echo "LoRA params: EPOCHS=$EPOCHS MAX_LENGTH=$MAX_LENGTH BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM=$GRAD_ACCUM LR=$LR WARMUP_RATIO=$WARMUP_RATIO LR_SCHEDULER_TYPE=$LR_SCHEDULER_TYPE WEIGHT_DECAY=$WEIGHT_DECAY SAVE_STEPS=$SAVE_STEPS LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA LORA_DROPOUT=$LORA_DROPOUT PRECISION=$PRECISION ENABLE_THINKING=$ENABLE_THINKING SAMPLE_LIMIT=$SAMPLE_LIMIT ATTN_IMPL=$ATTN_IMPL"

PREPARE_ARGS=(python "$SKILL_DIR/scripts/prepare_data.py" --output-dir "$DATA_DIR")
if [[ "$FORCE_DATA" == "1" ]]; then
  PREPARE_ARGS+=(--force)
fi
"${PREPARE_ARGS[@]}"

if [[ "$COMPLIANCE_CHECK" == "1" ]]; then
  python "$SKILL_DIR/scripts/check_competition_compliance.py" --model-id "$MODEL_ID"
fi

TRAIN_ARGS=(
  python "$SKILL_DIR/scripts/train_lora_baseline.py"
  --model-id "$MODEL_ID"
  --train-file "$DATA_DIR/train_all.jsonl"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LR"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --warmup-ratio "$WARMUP_RATIO"
  --lr-scheduler-type "$LR_SCHEDULER_TYPE"
  --weight-decay "$WEIGHT_DECAY"
  --save-steps "$SAVE_STEPS"
  --precision "$PRECISION"
  --adapter-base-model-name "OpenOneRec/OneReason-0.8B-pretrain-competition"
  --attn-impl "$ATTN_IMPL"
  --sample-limit "$SAMPLE_LIMIT"
)
if [[ "$ENABLE_THINKING" == "1" ]]; then
  TRAIN_ARGS+=(--enable-thinking)
fi
"${TRAIN_ARGS[@]}"

rm -rf "$UPLOAD_DIR"
mkdir -p "$UPLOAD_DIR"
cp "$OUTPUT_DIR/adapter_model.safetensors" "$UPLOAD_DIR/adapter_model.safetensors"
cp "$OUTPUT_DIR/adapter_config.json" "$UPLOAD_DIR/adapter_config.json"

python "$SKILL_DIR/scripts/validate_upload.py" \
  --method lora \
  --model-dir "$UPLOAD_DIR"

if [[ "$COMPLIANCE_CHECK" == "1" ]]; then
  python "$SKILL_DIR/scripts/check_competition_compliance.py" \
    --model-id "$MODEL_ID" \
    --adapter-dir "$UPLOAD_DIR"
fi

echo "Training output is in: $OUTPUT_DIR"
echo "Upload-ready LoRA files are in: $UPLOAD_DIR"
