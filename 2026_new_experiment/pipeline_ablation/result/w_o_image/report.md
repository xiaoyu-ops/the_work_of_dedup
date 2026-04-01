# Pipeline Report - 20260210-131036

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\w_o_image.yaml`
- Run ID: `20260210-131036`
- Stages completed: 3

## Overall Metrics
- Sorter inputs: 17,579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Modalities enabled: 2
- Modalities completed: 2
- Files processed across modalities: 7,705
- Modalities throughput: 8.08 files/s over 953.16s
- Sorter data throughput: 718.91 MB/s
- Aggregate dedup stats:
  - Total candidates: 11,045
  - Unique: 2,432
  - Duplicates: 4,908
  - Missing assets: 365
  - Copied artifacts: 0
  - Duplication rate: 44.44%
  - Unique ratio: 22.02%

## Sorter Summary
- Total inputs: 17579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260210-131036\stage1_sorter\manifest.csv`
- Elapsed: 3.48s
- Throughput: 5054.56 files/s
- Data throughput: 718.91 MB/s
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
- **stage1_sorter**: success (3.48s)
- **stage2_audio**: success (942.67s)
- **stage2_text**: success (10.48s)

## Modality Outcomes
- **audio** (status: completed)
  - Environment: audio
  - Files provided: 4365
  - Processed subset: 4365
  - Elapsed: 942.67s
  - Throughput: 4.63 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_image\audio`
  - Candidates: 4365.0
  - Stats: unique=2280.0, duplicates=1720.0, missing=365.0, copied=0.0
  - Duplication rate: 39.40%
  - Unique ratio: 52.23%
- **image** (status: disabled)
- **text** (status: completed)
  - Environment: text-dedup
  - Files provided: 3340
  - Processed subset: 3340
  - Elapsed: 10.48s
  - Throughput: 318.58 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_image\text`
  - Candidates: 6680.0
  - Stats: unique=152.0, duplicates=3188.0, missing=0.0, copied=0.0
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%