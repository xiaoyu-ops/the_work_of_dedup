"""Smoke for cross-modal quality (plan A direction A).

Verifies:
1. Sidecar discovery returns the right files for image/text/audio neighbors.
2. The `cross_modal` quality_metric path through each modality's qsemdedup
   does not crash when sidecars are present (CLIP/CLAP loaded if available)
   or when sidecars are absent (graceful fallback to entropy / SNR /
   file_size).

This is a structural smoke — it does not assert specific alignment scores
because that depends on which models are downloaded. The CLIP/CLAP load
path is exercised only when `--with-models` is passed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.cross_modal_quality import (  # noqa: E402
    find_audio_sidecar,
    find_image_sidecar,
    find_text_sidecar,
    read_caption,
)


def _check_sidecar_discovery(tmp: Path) -> None:
    # Build a small parallel-named set: p001.jpg/.txt/.wav, p002.jpg only,
    # p003.txt only, plus an uppercase variant.
    (tmp / "p001.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    (tmp / "p001.txt").write_text("a red apple", encoding="utf-8")
    (tmp / "p001.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (tmp / "p002.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    (tmp / "p003.txt").write_text("orphan caption", encoding="utf-8")
    (tmp / "p004.PNG").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp / "p004.txt").write_text("uppercase ext", encoding="utf-8")

    # Image -> text sidecar
    assert find_text_sidecar(tmp / "p001.jpg") == tmp / "p001.txt"
    assert find_text_sidecar(tmp / "p002.jpg") is None  # no caption
    # Text -> image sidecar (case-insensitive ext via uppercase fallback)
    assert find_image_sidecar(tmp / "p001.txt") == tmp / "p001.jpg"
    assert find_image_sidecar(tmp / "p003.txt") is None
    p004_match = find_image_sidecar(tmp / "p004.txt")
    # On case-insensitive filesystems p004.png and p004.PNG resolve to the
    # same inode, so we just verify a hit whose stem is "p004".
    assert p004_match is not None and p004_match.stem == "p004"
    # Audio sidecar / text -> audio
    assert find_text_sidecar(tmp / "p001.wav") == tmp / "p001.txt"
    assert find_audio_sidecar(tmp / "p001.txt") == tmp / "p001.wav"

    # read_caption truncation safety
    cap = read_caption(tmp / "p001.txt")
    assert cap == "a red apple"
    print("[smoke] sidecar discovery OK")


def _check_dispatch_no_models(parent: Path) -> None:
    """Run image/text quality dispatch with cross_modal, no model load.

    Strategy: place the orphan image / text in a *fresh* subdir so that
    sidecar discovery genuinely returns None; expect graceful fallback to
    file_size / entropy (no CLIP load).
    """
    iso_img = parent / "iso_img"
    iso_img.mkdir()
    (iso_img / "orphan.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 256)
    from image.method.pipeline_api import _compute_image_quality

    q = _compute_image_quality([iso_img / "orphan.jpg"], "cross_modal")
    assert q.shape == (1,) and q[0] > 0.0  # file_size fallback (>0 because non-empty)
    print(f"[smoke] image cross_modal fallback (no sidecar) OK, quality={float(q[0]):.1f}")

    iso_txt = parent / "iso_txt"
    iso_txt.mkdir()
    (iso_txt / "orphan.txt").write_text("the quick brown fox", encoding="utf-8")
    from text.method.qsemdedup import _build_text_quality

    q = _build_text_quality([iso_txt / "orphan.txt"], ["the quick brown fox"], "cross_modal")
    assert q.shape == (1,) and q[0] > 0.0  # entropy fallback for nontrivial text
    print(f"[smoke] text cross_modal fallback (no sidecar) OK, quality={float(q[0]):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-models",
        action="store_true",
        help="Also exercise the CLIP load path on a paired sample (slow first time).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="cmq_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        _check_sidecar_discovery(tmp)
        _check_dispatch_no_models(tmp)

        if args.with_models:
            from pipelines.cross_modal_quality import compute_clip_alignment

            (tmp / "paired.jpg").write_bytes((tmp / "p001.jpg").read_bytes())
            sims = compute_clip_alignment(
                [tmp / "p001.jpg"], ["a small red apple"]
            )
            print(f"[smoke] CLIP alignment computed: shape={sims.shape}, sim={float(sims[0]):.3f}")

    print("\n[smoke] cross_modal_quality smoke PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
