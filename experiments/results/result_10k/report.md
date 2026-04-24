# Pipeline Report - 20260130-113723

## Overview
- Config: `configs\my_pipeline_10k.yaml`
- Run ID: `20260130-113723`
- Stages completed: 2

## Overall Metrics
- Sorter inputs: 10,000
- Input volume: 965,818,956 bytes (921.08 MB)
- Modalities enabled: 1
- Modalities completed: 1
- Files processed across modalities: 10,000
- Modalities throughput: 83.27 files/s over 120.09s
- Sorter data throughput: 996.06 MB/s
- Aggregate dedup stats:
  - Total candidates: 10,000
  - Unique: 9,981
  - Duplicates: 19
  - Missing assets: 0
  - Copied artifacts: 1
  - Duplication rate: 0.19%
  - Unique ratio: 99.81%

## Sorter Summary
- Total inputs: 10000
- Input volume: 965,818,956 bytes (921.08 MB)
- Manifest: `D:\Deduplication_framework\2026_new_experiment\artifacts_10k\20260130-113723\stage1_sorter\manifest.csv`
- Elapsed: 0.92s
- Throughput: 10814.06 files/s
- Data throughput: 996.06 MB/s
- Sorter outcomes: success=10000, fail=0
- Move files enabled: False
- Per-modality counts:
  - audio: 0
  - image: 10000
  - text: 0
- Per-modality volume:
  - audio: 0 bytes (0.00 MB)
  - image: 965,818,956 bytes (921.08 MB)
  - text: 0 bytes (0.00 MB)

## Stage Summary
- **stage1_sorter**: success (0.92s)
- **stage2_image**: success (120.09s)

## Modality Outcomes
- **audio** (status: disabled)
- **image** (status: completed)
  - Environment: image
  - Files provided: 10000
  - Processed subset: 10000
  - Elapsed: 120.09s
  - Throughput: 83.27 files/s
  - Output dir: `D:\Deduplication_framework\2026_new_experiment\result_10k\image_processed`
  - Candidates: 10000.0
  - Stats: unique=9981.0, duplicates=19.0, missing=0.0, copied=1.0
  - Duplication rate: 0.19%
  - Unique ratio: 99.81%
- **text** (status: disabled)