"""Smoke for joint-similarity dedup (plan A direction B).

Synthesizes 8 pairs in a 2-modality setup (image + text) such that:
- p001 / p002 are near-duplicates in BOTH modalities  ⇒ should collapse
- p003 / p004 are near-dup in image only, distinct in text ⇒ should NOT collapse
- p005 / p006 are near-dup in text only, distinct in image ⇒ should NOT collapse
- p007 / p008 are distinct in both                  ⇒ should NOT collapse

Under the legacy "AND-of-per-modality" rule (current join_keepers), p001+p002
collapse, but ALSO p003+p004 collapse (image rule fires) AND p005+p006 collapse
(text rule fires) → the OR-shaped rule discards diverse pairs.

Under the new joint_similarity_dedup, only pairs near-duplicate in BOTH
modalities (p001+p002) collapse. The single-modality near-duplicates survive.

The smoke uses synthetic embeddings (no models loaded), so it runs in <1s.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pipelines.joint_dedup import joint_similarity_dedup  # noqa: E402


def _make_embedding(seed: int, dim: int = 64, near: int | None = None,
                    noise: float = 0.0) -> np.ndarray:
    """Return a deterministic dim-D vector. ``near`` reuses another seed's
    vector with optional Gaussian noise to model near-duplicates."""
    base_seed = near if near is not None else seed
    rng = np.random.default_rng(1000 + base_seed)
    v = rng.standard_normal(dim).astype(np.float32)
    if noise > 0:
        rng_noise = np.random.default_rng(2000 + seed)
        v = v + rng_noise.standard_normal(dim).astype(np.float32) * noise
    return v


def main() -> int:
    pair_ids = [f"p{i:03d}" for i in range(1, 9)]
    n = len(pair_ids)

    # Build per-modality embeddings.
    img = np.zeros((n, 64), dtype=np.float32)
    txt = np.zeros((n, 64), dtype=np.float32)

    # p001 / p002 — near-dup in both modalities
    img[0] = _make_embedding(seed=1)
    img[1] = _make_embedding(seed=10, near=1, noise=0.02)
    txt[0] = _make_embedding(seed=101)
    txt[1] = _make_embedding(seed=110, near=101, noise=0.02)

    # p003 / p004 — near-dup in image only
    img[2] = _make_embedding(seed=2)
    img[3] = _make_embedding(seed=20, near=2, noise=0.02)
    txt[2] = _make_embedding(seed=200)
    txt[3] = _make_embedding(seed=210)  # distinct seed → distinct text

    # p005 / p006 — near-dup in text only
    img[4] = _make_embedding(seed=3)
    img[5] = _make_embedding(seed=30)  # distinct image
    txt[4] = _make_embedding(seed=300)
    txt[5] = _make_embedding(seed=310, near=300, noise=0.02)

    # p007 / p008 — fully distinct
    img[6] = _make_embedding(seed=4)
    img[7] = _make_embedding(seed=40)
    txt[6] = _make_embedding(seed=400)
    txt[7] = _make_embedding(seed=410)

    result = joint_similarity_dedup(
        pair_ids,
        modality_embeddings={"image": img, "text": txt},
        modality_weights={"image": 0.5, "text": 0.5},
        eps=0.05,        # joint threshold = 0.95
        alpha=0.7,
        n_clusters=2,    # force a cluster so all 8 pairs are compared
    )

    print(f"[smoke] keepers ({len(result.pair_keepers)}): {sorted(result.pair_keepers)}")
    print(f"[smoke] drops:   {result.stats['joint_drops']}")
    for keeper, dups in result.dup_groups.items():
        for dup_id, sim in dups:
            print(f"  - {keeper} ← {dup_id}  sim_joint={sim:.3f}")

    keepers = set(result.pair_keepers)
    # Critical assertion: only one of p001/p002 survives (the joint dup).
    assert ("p001" in keepers) ^ ("p002" in keepers), (
        f"p001/p002 should collapse to a single survivor; got {keepers & {'p001','p002'}}"
    )
    # Single-modality near-dups must survive the joint-similarity rule.
    for pid in ("p003", "p004", "p005", "p006", "p007", "p008"):
        assert pid in keepers, f"{pid} should survive (only one modality is near-dup)"

    print("\n[smoke] joint_similarity_dedup PASSED — joint rule preserves single-modality diversity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
