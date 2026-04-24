# Pipeline Report - 20260225-100609

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\text_only.yaml`
- Run ID: `20260225-100609`
- Stages completed: 1

## Overall Metrics
- Sorter inputs: 17,579
- Modalities enabled: 1
- Modalities completed: 1
- Files processed across modalities: 3,340
- Modalities throughput: 216.80 files/s over 15.41s
- Aggregate dedup stats:
  - Total candidates: 6,680
  - Unique: 152
  - Duplicates: 3,188
  - Missing assets: 0
  - Copied artifacts: 152
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%

## Sorter Summary
- Total inputs: 17579
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260225-100609\stage1_sorter\manifest.csv`
- Per-modality counts:
  - audio: 4365
  - image: 7800
  - text: 3340
- Unknown categories:
  - unknown: 2074

## Stage Summary
- **stage2_text**: success (15.41s)

## Modality Outcomes
- **audio** (status: disabled)
- **image** (status: disabled)
- **text** (status: completed)
  - Environment: text-dedup
  - Files provided: 3340
  - Processed subset: 3340
  - Elapsed: 15.41s
  - Throughput: 216.80 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\text_only\text`
  - Candidates: 6680.0
  - Stats: unique=152.0, duplicates=3188.0, missing=0.0, copied=152.0
  - Duplication rate: 47.72%
  - Unique ratio: 2.28%