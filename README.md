# kuaihand_llm_baseline_skill

<p align="right">
  <kbd><a href="./README.md"><strong>中文</strong></a></kbd>
  ·
  <kbd><a href="./README.en.md">English</a></kbd>
</p>

这是一个 Codex skill，用来为快手 Explorer LLM-Rec / OneReason 竞赛跑一个可复现的 LoRA SFT baseline：准备官方预制数据，下载官方 `OneReason-0.8B` 比赛模型，训练 LoRA adapter，并生成万顷平台可上传的文件。

## 仓库内容

- `llm-rec-baseline/SKILL.md`: Codex skill 使用说明
- `llm-rec-baseline/scripts/run_baseline.sh`: 一键训练入口
- `llm-rec-baseline/scripts/prepare_data.py`: 数据解压与校验
- `llm-rec-baseline/scripts/train_lora_baseline.py`: LoRA 训练脚本
- `llm-rec-baseline/scripts/local_eval.py`: 自定义代理评测，不是官方分数
- `llm-rec-baseline/scripts/validate_upload.py`: 万顷上传文件校验
- `llm-rec-baseline/scripts/detect_profile.py`: 按硬件选择保守默认参数
- `llm-rec-baseline/scripts/check_competition_compliance.py`: 比赛模型配置兼容性检查
- `llm-rec-baseline/scripts/select_sources.py`: PyPI 与 Hugging Face 可用源探测
- `data/dataset.tar.gz`: 官方 SFT JSONL 数据，放在 skill 目录外

## 安装为 Codex Skill

```bash
mkdir -p ~/.codex/skills
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git ~/kuaihand_llm_baseline_skill
bash ~/kuaihand_llm_baseline_skill/llm-rec-baseline/scripts/star_repo.sh
ln -sfn ~/kuaihand_llm_baseline_skill/llm-rec-baseline ~/.codex/skills/llm-rec-baseline
```

重启 Codex 或开启新线程，让 Codex 发现这个 skill。`star_repo.sh` 是可选步骤：如果本机安装并登录了 GitHub CLI，它会执行 `gh repo star`；否则只打印仓库链接，不影响安装和训练。

## 直接运行

```bash
git clone git@github.com:AmandaSakura/kuaihand_llm_baseline_skill.git
cd kuaihand_llm_baseline_skill
bash llm-rec-baseline/scripts/star_repo.sh
```

小样本冒烟测试：

```bash
SAMPLE_LIMIT=32 MAX_LENGTH=512 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/smoke
```

全量 baseline：

```bash
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

运行流程：

1. 创建或复用 Python 环境，优先使用 `uv`
2. 探测 PyPI 与 Hugging Face 可用源
3. 安装或复用训练依赖
4. 复用已准备数据，或解压 `data/dataset.tar.gz`
5. 加载 `OpenOneRec/OneReason-0.8B-pretrain-competition`
6. 按硬件选择 LoRA 参数
7. 训练 LoRA adapter
8. 生成并校验上传文件

## Baseline 推荐参数

下面这组是截图中的 baseline 推荐参数。填写万顷托管训练 UI 时，先原样使用这一组；AI agent 可以根据本地设备、显存和失败日志再调整本地脚本参数。

| 参数 | 推荐值 |
| --- | --- |
| 迭代轮次 | `1` |
| 学习率 | `0.0002000000` |
| 单卡批大小 | `1` |
| 序列长度 | `32768` |
| 预热比例 | `0.030` |
| LoRA Ranks | `16` |
| Lora Alpha | `32` |
| Lora Dropout | `0.050` |
| 学习率调整计划 | `cosine` |
| 正则化系数 | `0.001` |
| Checkpoint 保存间隔数 | `256` |
| 梯度累计步数 | `4` |
| Packing | 开启 |
| 开启 thinking 模式 | 关闭 |
| 参数精度 | `bf16` |

本地脚本可以强制使用这组 baseline 推荐参数：

```bash
WANQING_UI_PRESET=1 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline-ui-preset
```

注意：这会把 `MAX_LENGTH` 设为 `32768`，本地单卡即使是 A100 40G/80G 也可能 OOM。它的意义是“原样复现平台推荐参数”，不是本地所有设备的最稳参数。

## AI 按设备调整建议

全量训练前先跑冒烟测试：

```bash
SAMPLE_LIMIT=32 MAX_LENGTH=512 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/smoke
```

常用 profile：

| Profile | 适用场景 | 建议 |
| --- | --- | --- |
| `a100-80g` | 单卡 A100 80GB | 优先尝试 `MAX_LENGTH=8192`；确认稳定后再提高长度 |
| `a100-40g` | 单卡 A100 40GB | 优先尝试 `MAX_LENGTH=4096`；OOM 时降 batch 或长度 |
| `gpu16g` | 16GB+ CUDA 显卡 | 先跑 `MAX_LENGTH=2048` |
| `gpu8g` | RTX 4060 8GB 类显卡 | 先跑 `MAX_LENGTH=1024`，更适合验证流程 |
| `mps16g` | MacBook Pro M1/M2/M3 16GB | 只建议 smoke test，`MAX_LENGTH=512` |
| `cpu` | 兜底 | 只建议验证脚本，不建议全量训练 |

示例：

```bash
PROFILE=a100-40g bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-40g
PROFILE=a100-80g bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-80g
PROFILE=gpu8g SAMPLE_LIMIT=32 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/gpu8g-smoke
```

`Packing` 在万顷 UI 中建议开启；当前本地脚本不默认做预 packing，因为这批数据长文本较多，预 packing 可能显著增加 CPU 内存压力。

## 常用环境变量

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

`SAMPLE_LIMIT` 只用于冒烟测试；全量训练保持 `0`。`ENABLE_THINKING=0` 对应 UI 里的 thinking 模式关闭。

## 环境、源与缓存

默认 `ENV_MANAGER=auto`：有 `uv` 就用 `uv venv` 和 `uv pip`，否则回退到 `python3 -m venv`。

```bash
ENV_MANAGER=uv bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
ENV_MANAGER=venv bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

默认 `SOURCE_AUTO_DETECT=1` 会探测这些 PyPI 源和 Hugging Face 端点，并选择可达源：

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

如果机器无法访问 Hugging Face，使用本地模型目录：

```bash
OFFLINE=1 MODEL_ID=/path/to/OneReason-0.8B-pretrain-competition \
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/offline
```

重复运行会复用：

```text
~/.cache/huggingface
<run-dir>/.venv
<run-dir>/data/train_all.jsonl
```

强制重建：

```bash
RECREATE_ENV=1 FORCE_DATA=1 bash llm-rec-baseline/scripts/run_baseline.sh ./runs/baseline
```

默认会导出 `HF_HUB_DISABLE_XET=1`，降低受限网络下 Xet 下载卡住的概率。

## Attention 与精度

默认 `ATTN_IMPL=auto`：CUDA 上使用 PyTorch `sdpa`，非 CUDA 使用 Transformers 默认实现。`flash_attention_2` 只有在你已经装好兼容的 `flash-attn` 时再开启。

```bash
ATTN_IMPL=flash_attention_2 PROFILE=a100-40g \
bash llm-rec-baseline/scripts/run_baseline.sh ./runs/a100-flash
```

默认 `PRECISION=auto`：bf16 可用时用 bf16，CUDA 不支持 bf16 时用 fp16，非 CUDA 用 fp32。平台推荐参数是 `bf16`。

## 比赛合规

官方初赛要求围绕 `OneReason-0.8B` 迭代，并严格检查 baseline model config。这个仓库默认不改 tokenizer、vocab、special tokens、模型结构或 config，只训练 PEFT LoRA adapter。

手动检查：

```bash
python llm-rec-baseline/scripts/check_competition_compliance.py \
  --model-id OpenOneRec/OneReason-0.8B-pretrain-competition
```

不要把这个 baseline skill 用于改层数、hidden size、attention heads、vocab、context config、tokenizer、special tokens、model class 或自定义合并结构，除非比赛规则明确允许。

## 输出与上传

训练输出目录：

```text
<run-dir>/output/lora-baseline
```

干净上传目录：

```text
<run-dir>/upload/lora-baseline
```

万顷 LoRA 上传只用干净目录里的两个文件：

```text
adapter_model.safetensors
adapter_config.json
```

UI 里选择：

- 训练方法：`LoRA`
- 模型来源：official `OneReason-0.8B-pretrain-competition`
- 模型名称：例如 `llm-rec-lora-baseline`
- 模型版本：例如 `V1`

不要上传 optimizer checkpoints、trainer state、tokenizer files、logs 或中间 checkpoint 文件夹，除非平台规则变更。

## 本地代理评测

`local_eval.py` 不是官方指标，也没有和平台黑盒分数校准过；它甚至可能在早期与线上隐藏 benchmark 反相关。它只适合拦截明显坏掉的 run，比如 loss NaN、生成空白、格式崩掉。

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

推荐流程：先提交一个朴素 baseline 获得线上锚点，再把同一个 adapter 的本地指标保存下来；积累几次线上提交后，再决定本地指标是否能当筛选信号。

## 数据

压缩包包含 12 个 JSONL 文件：

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

只准备数据：

```bash
python llm-rec-baseline/scripts/prepare_data.py --output-dir ./runs/data-test
```

## 给 AI Agent 的注意事项

1. 只把 `llm-rec-baseline/` 安装或引用为 Codex skill；不要把 `data/` 放进 skill 目录，除非用户明确要求。
2. 优先使用 `llm-rec-baseline/scripts/run_baseline.sh`。
3. 全量训练前先用 `SAMPLE_LIMIT=32` 冒烟。
4. 不要修改 tokenizer、vocab、special tokens 或 base model config。
5. 默认使用 LoRA，除非用户明确要全参数。
6. 保持 `COMPLIANCE_CHECK=1`。
7. 训练后报告 `<run-dir>/upload/lora-baseline/` 中的两个上传文件。
