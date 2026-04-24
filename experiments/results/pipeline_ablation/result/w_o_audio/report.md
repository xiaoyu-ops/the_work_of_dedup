# Pipeline Report - 20260210-132653

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\w_o_audio.yaml`
- Run ID: `20260210-132653`
- Stages completed: 3

## Overall Metrics
- Sorter inputs: 17,579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Modalities enabled: 2
- Modalities completed: 2
- Files processed across modalities: 11,140
- Modalities throughput: 96.11 files/s over 115.91s
- Sorter data throughput: 674.01 MB/s
- Aggregate dedup stats:
  - Total candidates: 14,480
  - Unique: 4,037
  - Duplicates: 7,103
  - Missing assets: 0
  - Copied artifacts: 34
  - Duplication rate: 49.05%
  - Unique ratio: 27.88%

## Sorter Summary
- Total inputs: 17579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260210-132653\stage1_sorter\manifest.csv`
- Elapsed: 3.71s
- Throughput: 4738.89 files/s
- Data throughput: 674.01 MB/s
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
- **stage1_sorter**: success (3.71s)
- **stage2_image**: success (105.69s)
- **stage2_text**: success (10.22s)

## Modality Outcomes
- **audio** (status: disabled)
- **image** (status: completed)
  - Environment: image
  - Files provided: 7800
  - Processed subset: 7800
  - Elapsed: 105.69s
  - Throughput: 73.80 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_audio\image`
  - Candidates: 7800.0
  - Stats: unique=3885.0, duplicates=3915.0, missing=0.0, copied=34.0
  - Duplication rate: 50.19%
  - Unique ratio: 49.81%
- **text** (status: completed)
  - Environment: text-dedup
  - Files provided: 3340
  - Processed subset: 3340
  - Elapsed: 10.22s
  - Throughput: 326.84 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_audio\text`
  - Candidates: 6680.0
  - Stats: unique=152.0, duplicates=3188.0, missing=0.0, copied=0.0
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%