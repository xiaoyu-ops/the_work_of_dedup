# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Dependencies are managed via `pyproject.toml` with [uv](https://github.com/astral-sh/uv).

```bash
# Install base only (sorter + orchestrator)
uv sync

# Install with specific modality extras
uv sync --extra image   # torch, open-clip-torch, scikit-learn, Pillow
uv sync --extra audio   # librosa, soundfile, scikit-image, imagehash
uv sync --extra text    # datasketch, simhash (default jaccard needs no extras)

# Install everything
uv sync --extra all
```

Text jaccard dedup requires no extras — only pyyaml (already in base).

## Running the Pipeline

```bash
# Run the full multimodal deduplication pipeline
python -m pipelines.multimodal_runner --config experiments/configs/my_pipeline.yaml

# Skip the report stage
python -m pipelines.multimodal_runner --config experiments/configs/my_pipeline.yaml --no-report

# Evaluate dedup results against ground truth
python experiments/scripts/evaluate_10k.py
python experiments/tools/run_comparative_evaluation.py
```

## Architecture Overview

The system is a three-stage pipeline orchestrated by `PipelineOrchestrator` (`pipelines/orchestrator.py`):

**Stage 1 – Sorter** (`pipelines/sorter.py` + `pipelines/sorter_stage.py`): Scans `general.input_root`, classifies every file into `image`, `audio`, or `text` by extension and content heuristics, then writes a `manifest.csv` to `artifacts/<run_id>/stage1_sorter/`. Files are strictly classified: only `.png/.jpg/.jpeg` for image, `.wav` for audio by default (`STRICT_IMAGE_EXTS`, `STRICT_AUDIO_EXTS` in `sorter.py`).

**Stage 2 – Modality Runners** (`pipelines/modalities/{image,audio,text}_runner.py`): Each runner is launched as a subprocess by the orchestrator. Communication happens entirely via environment variables:
- `PIPELINE_IMAGE_INPUT_LIST` / `PIPELINE_AUDIO_INPUT_LIST` / `PIPELINE_TEXT_INPUT_LIST` — path to a temp manifest file with one filepath per line
- `PIPELINE_IMAGE_OUTPUT_DIR` / `PIPELINE_AUDIO_OUTPUT_DIR` / `PIPELINE_TEXT_OUTPUT_DIR` — where to write results
- `PIPELINE_IMAGE_CONFIG_FILE` etc. — path to the modality-specific config YAML

Runners write a `{modality}_runner_summary.json` to their output dir; the orchestrator reads it back after the subprocess exits.

**Stage 3 – Report** (`orchestrator._generate_report_markdown`): Aggregates all runner summaries into `summary.json` and an optional Markdown report.

### Image Deduplication (Q-SemDeDup)

Implemented in `image/method/pipeline_api.py`. The core innovation is quality-aware selection within each cluster:

```
Score = α · Sim(x, C) + (1-α) · Norm(Quality(x))     [α = 0.7 by default]
```

Backed by `open_clip` (CLIP-ViT-B-16) with a fallback to `average_rgb`. Configured via `experiments/configs/image_config.yaml` — key fields: `embedding.backend`, `embedding.model_name`, `dedup.method` (`semdedup` or `pairwise`), `dedup.eps`.

### Audio Deduplication

`audio/method/pipeline_api.py` — fingerprint-based dedup using `spectrum_fingerprint.py` (librosa + scikit-image). Precomputed fingerprints can be supplied via config to skip heavy computation. LSH dedup is handled via `LSH_deal_with_photo.py`.

### Text Deduplication

`text/method/pipeline_api.py` — n-gram Jaccard similarity. Configured via `experiments/configs/text_override.yaml`. Falls back to a sliding-window comparison when candidate count exceeds `max_candidates`.

## Configuration

The pipeline is driven by a single YAML/JSON config file. Key `general` fields:

| Field | Effect |
|---|---|
| `input_root` | Directory to scan for input files |
| `output_root` | Where final results and reports are written |
| `temp_root` | Artifacts directory (logs, manifests, stage flags) |
| `resume: true` | Skip stages whose config hash already has a `_SUCCESS` flag |
| `parallel_modalities: true` | Run image/audio/text stages concurrently via `ThreadPoolExecutor` |
| `batch_size` | How many files per orchestrator chunk (use `0` to disable chunking for image) |

Each modality section requires `enabled`, `entrypoint` (absolute path to runner script), `output_dir`, and optionally `config_file`, `env` (extra env vars), `workdir`, `max_workers`.

When using a single uv environment for all modalities, leave `executor.envs` empty or point all stages to the same env. The executor will use the current Python interpreter directly when no conda env is specified.

## Stage Flags and Artifacts

The orchestrator writes `_SUCCESS`, `_FAILURE`, and `_LOCK` sentinel files inside each stage's artifact directory (`artifacts/<run_id>/stage1_sorter/`, `stage2_image/`, etc.). A stale `_LOCK` will abort the run — delete it manually before retrying. `_FAILURE` is preserved across runs to warn on retry.

Each successful modality stage also saves per-chunk `stdout_N.log` / `stderr_N.log` and a copy of the runner's duplicates file.

## Requirements

All dependencies are declared in `pyproject.toml` as optional extras (`image`, `audio`, `text`, `all`). See `requirements/README.md` for install commands. The old per-modality `requirements/*.txt` files (Windows-only) have been removed.

## Project Structure

```
the_work_of_dedup/
├── paper/                          # 论文相关
│   ├── latex/                      # LaTeX 源码（main.tex, references.bib, figures/）
│   ├── submission.pdf              # 投稿 PDF（VLDB 2026，已拒稿）
│   ├── review.pdf                  # 原始审稿意见 PDF
│   ├── meta-review.pdf             # Meta-review PDF
│   └── 审稿意见.md                  # 中文审稿意见详细分析
│
├── pipelines/                      # 流水线编排（orchestrator、modality runners）
│   └── sorter.py                   # Stage 1：文件分类器
├── image/                          # 图像去重模块
├── audio/                          # 音频去重模块
├── text/                           # 文本去重模块
│
├── experiments/                    # 实验相关
│   ├── configs/                    # Pipeline 配置 YAML
│   ├── scripts/                    # 实验脚本
│   │   ├── run_full_experiment.py  # 端到端实验 runner
│   │   ├── insert_duplicates.py    # 注入合成重复数据
│   │   └── evaluate_10k.py         # 10k 评测
│   ├── tools/                      # 辅助工具
│   │   ├── generate_mix_dataset_10k.py
│   │   └── run_comparative_evaluation.py  # 对比 SemDeDup / SimCLR
│   └── results/                    # 实验结果（ablation、对比实验、2026 数据）
│
├── docs/                           # 设计文档
├── requirements/                   # 依赖说明
└── pyproject.toml
```

> **注意**：代码目录（`pipelines/`, `image/`, `audio/`, `text/`, `sorter.py`）保留在根目录以维持 Python 导入路径（`python -m pipelines.multimodal_runner`）。
