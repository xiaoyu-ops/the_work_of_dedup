# Environment Setup

Dependencies are managed via `pyproject.toml` in the project root using [uv](https://github.com/astral-sh/uv).

## Quick Start

```bash
# Install base deps only (sorter + orchestrator)
uv sync

# Install with a specific modality
uv sync --extra image
uv sync --extra audio
uv sync --extra text

# Install everything
uv sync --extra all
```

## Extras

| Extra | Key packages | Use case |
|---|---|---|
| `image` | torch, open-clip-torch, scikit-learn, Pillow | CLIP embedding + SemDeDup |
| `audio` | librosa, soundfile, scikit-image, imagehash | Spectrogram fingerprinting |
| `text` | datasketch, simhash | MinHash LSH / SimHash (optional; default jaccard needs no extras) |

## Pipeline Config

When running all modalities in one environment, set `executor.envs` to the same env name for all stages, or leave empty to use the current Python interpreter.
