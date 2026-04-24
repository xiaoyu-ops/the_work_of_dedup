# Pipeline Report - 20260210-125026

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\w_o_sorter.yaml`
- Run ID: `20260210-125026`
- Stages completed: 3

## Overall Metrics
- Sorter inputs: 17,579
- Modalities enabled: 3
- Modalities completed: 3
- Files processed across modalities: 15,505
- Modalities throughput: 14.66 files/s over 1057.50s
- Aggregate dedup stats:
  - Total candidates: 18,845
  - Unique: 6,442
  - Duplicates: 8,698
  - Missing assets: 365
  - Copied artifacts: 50
  - Duplication rate: 46.16%
  - Unique ratio: 34.18%

## Sorter Summary
- Total inputs: 17579
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260210-125026\stage1_sorter\manifest.csv`
- Per-modality counts:
  - audio: 4365
  - image: 7800
  - text: 3340
- Unknown categories:
  - unknown: 2074

## Stage Summary
- **stage2_image**: success (105.64s)
- **stage2_audio**: success (941.94s)
- **stage2_text**: success (9.92s)

## Modality Outcomes
- **audio** (status: completed)
  - Environment: audio
  - Files provided: 4365
  - Processed subset: 4365
  - Elapsed: 941.94s
  - Throughput: 4.63 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_sorter\audio`
  - Candidates: 4365.0
  - Stats: unique=2259.0, duplicates=1741.0, missing=365.0, copied=0.0
  - Duplication rate: 39.89%
  - Unique ratio: 51.75%
- **image** (status: completed)
  - Environment: image
  - Files provided: 7800
  - Processed subset: 7800
  - Elapsed: 105.64s
  - Throughput: 73.83 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_sorter\image`
  - Candidates: 7800.0
  - Stats: unique=4031.0, duplicates=3769.0, missing=0.0, copied=50.0
  - Duplication rate: 48.32%
  - Unique ratio: 51.68%
- **text** (status: completed)
  - Environment: text-dedup
  - Files provided: 3340
  - Processed subset: 3340
  - Elapsed: 9.92s
  - Throughput: 336.63 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\w_o_sorter\text`
  - Candidates: 6680.0
  - Stats: unique=152.0, duplicates=3188.0, missing=0.0, copied=0.0
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%