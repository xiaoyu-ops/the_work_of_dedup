# Pipeline Report - 20260210-123229

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\full.yaml`
- Run ID: `20260210-123229`
- Stages completed: 4

## Overall Metrics
- Sorter inputs: 17,579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Modalities enabled: 3
- Modalities completed: 3
- Files processed across modalities: 15,505
- Modalities throughput: 14.73 files/s over 1052.47s
- Sorter data throughput: 707.58 MB/s
- Aggregate dedup stats:
  - Total candidates: 18,845
  - Unique: 6,447
  - Duplicates: 8,693
  - Missing assets: 365
  - Copied artifacts: 50
  - Duplication rate: 46.13%
  - Unique ratio: 34.21%

## Sorter Summary
- Total inputs: 17579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260210-123229\stage1_sorter\manifest.csv`
- Elapsed: 3.53s
- Throughput: 4974.92 files/s
- Data throughput: 707.58 MB/s
- Sorter outcomes: success=15505, fail=2074
- Move files enabled: False
- Per-modality counts:
  - audio: 4365
  - image: 7800
  - text: 3340
- Per-modality volume:
  - audio: 1,764,237,320 bytes (1682.51 MB)
  - image: 852,487,605 bytes (813.00 MB)
  - text: 572,681 bytes (0.55 MB)
- Unknown categories:
  - unknown: 2074
- Unknown volume:
  - unknown: 4,417,550 bytes (4.21 MB)

## Stage Summary
- **stage1_sorter**: success (3.53s)
- **stage2_image**: success (104.36s)
- **stage2_audio**: success (937.47s)
- **stage2_text**: success (10.64s)

## Modality Outcomes
- **audio** (status: completed)
  - Environment: audio
  - Files provided: 4365
  - Processed subset: 4365
  - Elapsed: 937.47s
  - Throughput: 4.66 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\full\audio`
  - Candidates: 4365.0
  - Stats: unique=2264.0, duplicates=1736.0, missing=365.0, copied=0.0
  - Duplication rate: 39.77%
  - Unique ratio: 51.87%
- **image** (status: completed)
  - Environment: image
  - Files provided: 7800
  - Processed subset: 7800
  - Elapsed: 104.36s
  - Throughput: 74.74 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\full\image`
  - Candidates: 7800.0
  - Stats: unique=4031.0, duplicates=3769.0, missing=0.0, copied=50.0
  - Duplication rate: 48.32%
  - Unique ratio: 51.68%
- **text** (status: completed)
  - Environment: text-dedup
  - Files provided: 3340
  - Processed subset: 3340
  - Elapsed: 10.64s
  - Throughput: 313.91 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\full\text`
  - Candidates: 6680.0
  - Stats: unique=152.0, duplicates=3188.0, missing=0.0, copied=0.0
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%