# Pipeline Report - 20260222-083734

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\w_o_text.yaml`
- Run ID: `20260222-083734`
- Stages completed: 3

## Overall Metrics
- Sorter inputs: 17,579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Modalities enabled: 2
- Modalities completed: 2
- Files processed across modalities: 12,165
- Modalities throughput: 9.96 files/s over 1221.48s
- Sorter data throughput: 306.15 MB/s
- Aggregate dedup stats:
  - Total candidates: 12,165
  - Unique: 6,345
  - Duplicates: 5,455
  - Missing assets: 365
  - Copied artifacts: 68
  - Duplication rate: 44.84%
  - Unique ratio: 52.16%

## Sorter Summary
- Total inputs: 17579
- Input volume: 2,621,715,156 bytes (2500.26 MB)
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260222-083734\stage1_sorter\manifest.csv`
- Elapsed: 8.17s
- Throughput: 2152.46 files/s
- Data throughput: 306.15 MB/s
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
- **stage1_sorter**: success (8.17s)
- **stage2_image**: success (130.61s)
- **stage2_audio**: success (1090.88s)

## Modality Outcomes
- **audio** (status: completed)
  - Environment: audio
  - Files provided: 4365
  - Processed subset: 4365
  - Elapsed: 1090.88s
  - Throughput: 4.00 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_text\audio`
  - Candidates: 4365.0
  - Stats: unique=2263.0, duplicates=1737.0, missing=365.0, copied=0.0
  - Duplication rate: 39.79%
  - Unique ratio: 51.84%
- **image** (status: completed)
  - Environment: image
  - Files provided: 7800
  - Processed subset: 7800
  - Elapsed: 130.61s
  - Throughput: 59.72 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_text\image`
  - Candidates: 7800.0
  - Stats: unique=4082.0, duplicates=3718.0, missing=0.0, copied=68.0
  - Duplication rate: 47.67%
  - Unique ratio: 52.33%
- **text** (status: disabled)