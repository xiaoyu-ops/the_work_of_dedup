"""Generate a tiny synthetic multimodal corpus for end-to-end smokes.

Layout:
    <root>/dataset/p<id>.jpg
    <root>/dataset/p<id>.txt

Default root is ``~/.cache/the_work_of_dedup/mini`` to match
``experiments/configs/my_pipeline_local.yaml``. Pass ``--root`` to override.

Composition (10 image-text pairs total):
    p001..p005  unique images + unique captions
    p006..p008  near-duplicate images (small noise overlay) of p001..p003,
                with paraphrased captions of p001..p003
    p009..p010  exact-duplicate captions of p004..p005, but unique images

This gives the orchestrator something interesting to dedup in every modality
without depending on any external dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

try:
    from PIL import Image  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required to generate mini images. Run `uv sync --extra image`."
    ) from exc


PAIRS: list[tuple[str, tuple[int, int, int], str]] = [
    # (pair_id, base_color (R,G,B), caption)
    ("p001", (220, 60, 60), "A bright red apple resting on a wooden table."),
    ("p002", (60, 160, 220), "A clear blue ocean stretching to the horizon."),
    ("p003", (40, 180, 80), "Lush green forest with sunlight piercing the canopy."),
    ("p004", (240, 200, 60), "A field of yellow sunflowers under a summer sky."),
    ("p005", (180, 80, 200), "A vivid purple flower in close-up macro shot."),
]

# Near-image / paraphrase-text duplicates of the first three.
NEAR: list[tuple[str, str, str]] = [
    ("p006", "p001", "On a wooden surface lies one shiny red apple."),
    ("p007", "p002", "The blue ocean extends out to the distant horizon."),
    ("p008", "p003", "A green forest where sunbeams filter through the leaves."),
]

# Exact-text duplicates of p004 / p005, but with their own (unique) images.
EXACT_TEXT: list[tuple[str, str]] = [
    ("p009", "p004"),
    ("p010", "p005"),
]


def _make_image(color: tuple[int, int, int], *, noise: float = 0.0, seed: int = 0) -> Image.Image:
    """Build a 224x224 RGB image of the requested base color, plus optional noise."""
    rng = np.random.default_rng(seed)
    arr = np.zeros((224, 224, 3), dtype=np.float32)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    if noise > 0:
        arr += rng.normal(0.0, noise * 255.0, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate(root: Path) -> dict[str, int]:
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    color_lookup = {pid: color for pid, color, _ in PAIRS}
    text_lookup = {pid: cap for pid, _, cap in PAIRS}

    written_images = 0
    written_texts = 0

    # Originals
    for i, (pid, color, caption) in enumerate(PAIRS):
        img = _make_image(color, noise=0.0, seed=i)
        img.save(dataset / f"{pid}.jpg", quality=92)
        written_images += 1
        (dataset / f"{pid}.txt").write_text(caption + "\n", encoding="utf-8")
        written_texts += 1

    # Near-image, paraphrased-text duplicates
    for i, (pid, src, caption) in enumerate(NEAR):
        img = _make_image(color_lookup[src], noise=0.02, seed=100 + i)
        img.save(dataset / f"{pid}.jpg", quality=92)
        written_images += 1
        (dataset / f"{pid}.txt").write_text(caption + "\n", encoding="utf-8")
        written_texts += 1

    # Exact-text duplicates with unique images (use distinct random colors)
    rng = np.random.default_rng(999)
    for i, (pid, text_src) in enumerate(EXACT_TEXT):
        color = tuple(int(c) for c in rng.integers(40, 200, size=3))
        img = _make_image(color, noise=0.0, seed=200 + i)
        img.save(dataset / f"{pid}.jpg", quality=92)
        written_images += 1
        (dataset / f"{pid}.txt").write_text(text_lookup[text_src] + "\n", encoding="utf-8")
        written_texts += 1

    return {
        "dataset_dir": str(dataset),
        "images": written_images,
        "texts": written_texts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/.cache/the_work_of_dedup/mini").expanduser(),
        help="Root directory for the mini corpus (default: ~/.cache/the_work_of_dedup/mini)",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    info = generate(args.root)
    print(
        f"[mini] wrote {info['images']} images and {info['texts']} captions to {info['dataset_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
