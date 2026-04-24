# Pipeline Report - 20260225-094904

## Overview
- Config: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\configs\audio_only.yaml`
- Run ID: `20260225-094904`
- Stages completed: 1

## Overall Metrics
- Sorter inputs: 17,579
- Modalities enabled: 1
- Modalities completed: 1
- Files processed across modalities: 4,365
- Modalities throughput: 4.26 files/s over 1024.48s
- Aggregate dedup stats:
  - Total candidates: 4,365
  - Unique: 2,262
  - Duplicates: 1,738
  - Missing assets: 365
  - Copied artifacts: 2,262
  - Duplication rate: 39.82%
  - Unique ratio: 51.82%

## Sorter Summary
- Total inputs: 17579
- Manifest: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\artifacts\20260225-094904\stage1_sorter\manifest.csv`
- Per-modality counts:
  - audio: 4365
  - image: 7800
  - text: 3340
- Unknown categories:
  - unknown: 2074

## Stage Summary
- **stage2_audio**: success (1024.48s)

## Modality Outcomes
- **audio** (status: completed)
  - Environment: audio
  - Files provided: 4365
  - Processed subset: 4365
  - Elapsed: 1024.48s
  - Throughput: 4.26 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\result\audio_only\audio`
  - Candidates: 4365.0
  - Stats: unique=2262.0, duplicates=1738.0, missing=365.0, copied=2262.0
  - Duplication rate: 39.82%
  - Unique ratio: 51.82%
- **image** (status: disabled)
- **text** (status: disabled)