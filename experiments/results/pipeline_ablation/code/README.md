# Pipeline Ablation Runner

This folder contains the one-click runner that prepares configs and executes six pipeline ablations on `mix_dataset`.

## What it does
- Generates per-run pipeline configs under `code/configs/`
- Executes 6 ablation runs via `python -m pipelines`
- Writes per-run outputs to `pipeline_ablation/result/<run_name>/`
- Writes a consolidated summary to:
  - `pipeline_ablation/result/ablation_summary.csv`
  - `pipeline_ablation/result/ablation_summary.md`

## Input dataset
- Default: `D:\Deduplication_framework\mix_dataset`
- Override with `ABLATION_INPUT_ROOT`

## Runs
1. Full Pipeline (sorter+image+audio+text)
2. w/o Sorter (reuses previous manifest)
3. w/o Near-Dedup (image eps=0, audio/text md5)
4. w/o Image Dedup
5. w/o Audio Dedup
6. w/o Text Dedup

## Execute
```powershell
python D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\run_pipeline_ablation.py
```

## Storage save
- Computed as the sum of byte sizes of deduplicated files based on stage duplicate lists and the sorter manifest.

## Missing counts
- `image_missing`, `audio_missing`, `text_missing` are included to explain low overall dedup rates when many files are invalid.

## Rebuild summary only (no reruns)
```powershell
python D:\Deduplication_framework\2026_new_experiment\pipeline_ablation\code\run_pipeline_ablation.py --summary-only
```

## Virtual env config
- The script reads `executor.conda_executable` and `executor.envs` from `configs/my_pipeline_smoke.yaml` by default.
- Override base config path with:
  - `ABLATION_BASE_CONFIG=D:\Deduplication_framework\configs\my_pipeline.yaml`
