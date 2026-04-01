# Pipeline Report - 20260210-130805

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\w_o_near_dedup.yaml`
- Run ID: `20260210-130805`
- Stages completed: 4

## Overall Metrics
- Sorter inputs: 17,579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Modalities enabled: 3
- Modalities completed: 3
- Files processed across modalities: 15,505
- Modalities throughput: 122.24 files/s over 126.84s
- Sorter data throughput: 720.46 MB/s
- Aggregate dedup stats:
  - Total candidates: 18,845
  - Unique: 10,696
  - Duplicates: 4,809
  - Missing assets: 0
  - Copied artifacts: 0
  - Duplication rate: 25.52%
  - Unique ratio: 56.76%

## Sorter Summary
- Total inputs: 17579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260210-130805\stage1_sorter\manifest.csv`
- Elapsed: 3.47s
- Throughput: 5065.47 files/s
- Data throughput: 720.46 MB/s
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
- **stage1_sorter**: success (3.47s)
- **stage2_image**: success (111.24s)
- **stage2_audio**: success (12.17s)
- **stage2_text**: success (3.44s)

## Modality Outcomes
- **audio** (status: completed)
  - Environment: audio
  - Files provided: 4365
  - Processed subset: 4365
  - Elapsed: 12.17s
  - Throughput: 358.61 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_near_dedup\audio`
  - Candidates: 4365.0
  - Stats: unique=2744.0, duplicates=1621.0, missing=0.0, copied=0.0
  - Duplication rate: 37.14%
  - Unique ratio: 62.86%
- **image** (status: completed)
  - Environment: image
  - Files provided: 7800
  - Processed subset: 7800
  - Elapsed: 111.24s
  - Throughput: 70.12 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_near_dedup\image`
  - Candidates: 7800.0
  - Stats: unique=7800.0, duplicates=0.0, missing=0.0, copied=0.0
  - Duplication rate: 0.00%
  - Unique ratio: 100.00%
- **text** (status: completed)
  - Environment: text-dedup
  - Files provided: 3340
  - Processed subset: 3340
  - Elapsed: 3.44s
  - Throughput: 971.78 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_near_dedup\text`
  - Candidates: 6680.0
  - Stats: unique=152.0, duplicates=3188.0, missing=0.0, copied=0.0
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%