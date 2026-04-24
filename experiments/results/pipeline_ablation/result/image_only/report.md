# Pipeline Report - 20260225-094713

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\image_only.yaml`
- Run ID: `20260225-094713`
- Stages completed: 1

## Overall Metrics
- Sorter inputs: 17,579
- Modalities enabled: 1
- Modalities completed: 1
- Files processed across modalities: 7,800
- Modalities throughput: 70.98 files/s over 109.89s
- Aggregate dedup stats:
  - Total candidates: 7,800
  - Unique: 4,082
  - Duplicates: 3,718
  - Missing assets: 0
  - Copied artifacts: 0
  - Duplication rate: 47.67%
  - Unique ratio: 52.33%

## Sorter Summary
- Total inputs: 17579
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260225-094713\stage1_sorter\manifest.csv`
- Per-modality counts:
  - audio: 4365
  - image: 7800
  - text: 3340
- Unknown categories:
  - unknown: 2074

## Stage Summary
- **stage2_image**: success (109.89s)

## Modality Outcomes
- **audio** (status: disabled)
- **image** (status: completed)
  - Environment: image
  - Files provided: 7800
  - Processed subset: 7800
  - Elapsed: 109.89s
  - Throughput: 70.98 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\image_only\image`
  - Candidates: 7800.0
  - Stats: unique=4082.0, duplicates=3718.0, missing=0.0, copied=0.0
  - Duplication rate: 47.67%
  - Unique ratio: 52.33%
- **text** (status: disabled)