# MMdedup: A Parallel and Pipelined Multimodal Data Deduplication Framework

> **Paper**: *MMdedup: A Parallel and Pipelined Multimodal Data Deduplication Framework for MLLM Training*
> Submitted to VLDB 2026 Research Track (#2442)

A high-performance, modular framework for deduplicating massive multi-modal datasets (Image, Audio, Text) designed for MLLM training data pipelines.

## Key Innovation: Quality-Aware Semantic Deduplication (Q-SemDeDup)

Unlike traditional methods (e.g., SemDeDup) that blindly select the image closest to the cluster centroid, our framework introduces **Quality-Aware Sorting**:

$$Score = \alpha \cdot Sim(x, C) + (1-\alpha) \cdot Norm(Quality(x))$$

- $Sim(x, C)$: semantic similarity to the cluster centroid (representativeness)
- $Quality(x)$: image quality metric (e.g., file size / resolution)
- $\alpha = 0.7$ by default — balances semantic precision with visual quality

This retains the **highest-quality version** of an image among near-duplicates, acting as both a deduplicator and a dataset cleaner.

## Features

- **Automated Zero-Shot SemDeDup**: runs `MiniBatchKMeans` on-the-fly; no pre-computed K-Means indices required. Falls back to folder-based strategy when memory is constrained.
- **High Throughput**: >100 files/s on standard hardware (3–5× faster than baseline SemDeDup).
- **Multi-Modal Pipeline**: Image (CLIP/Q-SemDeDup), Audio (spectrum fingerprinting + LSH), Text (n-gram Jaccard / MinHash).
- **Resume Support**: stage-level `_SUCCESS` flags skip already-completed stages on retry.
- **Parallel Modalities**: image/audio/text stages run concurrently via `ThreadPoolExecutor`.

## Architecture

Three-stage pipeline orchestrated by `PipelineOrchestrator`:

```
Input Files
    │
    ▼
Stage 1 — Sorter          pipelines/sorter.py
    │  Classifies files into image / audio / text by extension + heuristics
    │  Writes manifest.csv to artifacts/<run_id>/stage1_sorter/
    ▼
Stage 2 — Modality Runners (parallel)
    ├── image/method/pipeline_api.py   CLIP embeddings → Q-SemDeDup clustering
    ├── audio/method/pipeline_api.py   Spectrum fingerprint → LSH dedup
    └── text/method/pipeline_api.py    N-gram Jaccard similarity
    │  Each runner writes {modality}_runner_summary.json to its output dir
    ▼
Stage 3 — Report           orchestrator._generate_report_markdown
    Aggregates all summaries → summary.json + Markdown report
```

## Benchmark Results

### Full-Scale Run (3.8M images — ImageNet-bloated)

| Metric | Value |
| :--- | :--- |
| Total inputs | 3,828,733 |
| Throughput | **101.22 files/s** |
| Duplicates removed | 2,574,850 (67.25%) |
| Unique retained | 1,253,883 (32.75%) |

### Ablation Study (17.5k mixed multimodal dataset)

| Configuration | Dedup Rate | Storage Saved |
| :--- | :--- | :--- |
| Image dedup only | 47.67% | 200.47 MB |
| Audio dedup only | 39.82% | 731.02 MB |
| Text dedup only | 47.72% | 0.51 MB |

### Image Dedup Comparison (10k subset)

| Method | Precision | Recall | Throughput | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Ours (Q-SemDeDup)** | **85.2%** | 69–90% | **~100 imgs/s** | Quality-aware selection |
| SemDeDup (original) | 93.7% | 96.2% | 27.9 imgs/s | Static, pre-computed only |
| SimCLR | 18.2% | 99.6% | 45.0 imgs/s | Low precision |

## Usage

### 1. Install

```bash
# Base only (sorter + orchestrator)
uv sync

# With image support (CLIP, scikit-learn, Pillow)
uv sync --extra image

# Everything
uv sync --extra all
```

### 2. Configure

Edit `experiments/configs/my_pipeline.yaml` to point to your dataset.

### 3. Run

```bash
# Full multimodal pipeline
python -m pipelines.multimodal_runner --config experiments/configs/my_pipeline.yaml

# Skip the report stage
python -m pipelines.multimodal_runner --config experiments/configs/my_pipeline.yaml --no-report

# Evaluate against ground truth
python experiments/scripts/evaluate_10k.py
python experiments/tools/run_comparative_evaluation.py
```

## Project Structure

```
the_work_of_dedup/
├── paper/                          # Paper assets
│   ├── latex/                      # LaTeX source (main.tex, references.bib, figures/)
│   ├── submission.pdf              # Submitted PDF (VLDB 2026)
│   ├── review.pdf                  # Reviewer comments PDF
│   ├── meta-review.pdf             # Meta-review PDF
│   └── 审稿意见.md                  # Detailed review analysis (Chinese)
│
├── pipelines/                      # Pipeline orchestration
│   ├── orchestrator.py             # Main 3-stage orchestrator
│   ├── sorter.py                   # Stage 1: file classifier
│   ├── sorter_stage.py
│   └── modalities/                 # Per-modality subprocess runners
│
├── image/                          # Image dedup module (Q-SemDeDup)
├── audio/                          # Audio dedup module (fingerprint + LSH)
├── text/                           # Text dedup module (n-gram Jaccard)
│
├── experiments/
│   ├── configs/                    # Pipeline YAML configs
│   ├── scripts/                    # Experiment scripts
│   │   ├── run_full_experiment.py
│   │   ├── insert_duplicates.py    # Synthetic duplicate injection
│   │   └── evaluate_10k.py
│   ├── tools/                      # Comparative evaluation helpers
│   │   ├── generate_mix_dataset_10k.py
│   │   └── run_comparative_evaluation.py
│   └── results/                    # Experiment outputs (ablation, comparisons)
│
├── docs/                           # Design documents
├── pyproject.toml                  # Dependencies (uv extras: image, audio, text, all)
└── CLAUDE.md                       # Dev guide for Claude Code
```

## Configuration Reference

Key `general` fields in the pipeline YAML:

| Field | Effect |
| :--- | :--- |
| `input_root` | Directory to scan for input files |
| `output_root` | Where final results and reports are written |
| `temp_root` | Artifacts directory (logs, manifests, stage flags) |
| `resume: true` | Skip stages with an existing `_SUCCESS` flag |
| `parallel_modalities: true` | Run image/audio/text stages concurrently |
| `batch_size` | Files per orchestrator chunk (`0` disables chunking for image) |

---

*For detailed documentation, see [docs/README.md](docs/README.md).*
