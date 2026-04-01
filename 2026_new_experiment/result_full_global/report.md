# Pipeline Report - 20260131-163924

## Overview
- Config: `configs\my_pipeline_full.yaml`
- Run ID: `20260131-163924`
- Stages completed: 1

## Overall Metrics
- Sorter inputs: 3,828,733
- Modalities enabled: 1
- Modalities completed: 1
- Files processed across modalities: 3,828,733
- Modalities throughput: 101.22 files/s over 37827.27s
- Aggregate dedup stats:
  - Total candidates: 3,828,733
  - Unique: 1,253,883
  - Duplicates: 2,574,850
  - Missing assets: 0
  - Copied artifacts: 191,218
  - Duplication rate: 67.25%
  - Unique ratio: 32.75%

## Sorter Summary
- Total inputs: 3828733
- Manifest: `D:\Deduplication_framework\2026_new_experiment\artifacts_full_global\20260131-163924\stage1_sorter\manifest.csv`
- Per-modality counts:
  - audio: 0
  - image: 3828733
  - text: 0

## Stage Summary
- **stage2_image**: success (37827.27s)

## Modality Outcomes
- **audio** (status: disabled)
- **image** (status: completed)
  - Environment: image
  - Files provided: 3828733
  - Processed subset: 3828733
  - Elapsed: 37827.27s
  - Throughput: 101.22 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\result_full_global\image_processed`
  - Candidates: 3828733.0
  - Stats: unique=1253883.0, duplicates=2574850.0, missing=0.0, copied=191218.0
  - Duplication rate: 67.25%
  - Unique ratio: 32.75%
- **text** (status: disabled)