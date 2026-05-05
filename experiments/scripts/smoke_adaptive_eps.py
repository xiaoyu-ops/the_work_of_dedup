"""Smoke for adaptive ε (plan A direction C).

Two synthetic clusters of 4 pairs each:
- TIGHT cluster: all 4 pairs have nearly-identical embeddings → high
  compactness → adaptive ε should LOOSEN threshold (lower) → more drops.
- LOOSE cluster: 4 distinct pairs sharing a coarse direction → low
  compactness → adaptive ε should TIGHTEN threshold (higher) → fewer drops.

Compares static-ε vs adaptive-ε behavior on the same 8-pair input. Asserts:
- adaptive_eps_lambda=0  -> behaves identically to plain joint_similarity_dedup
- adaptive_eps_lambda>0  -> per-cluster thresholds reported in stats reflect
  the compactness gradient (loose cluster threshold > tight cluster threshold)

Synthetic embeddings only; runs in <1s.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from pipelines.joint_dedup import joint_similarity_dedup  # noqa: E402
from pipelines.qsemdedup_core import adaptive_threshold, cluster_compactness  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(42)
    dim = 32

    # Tight cluster: a base vector + tiny noise
    base_tight = rng.standard_normal(dim).astype(np.float32)
    tight = np.stack(
        [base_tight + rng.standard_normal(dim).astype(np.float32) * 0.005 for _ in range(4)],
        axis=0,
    )

    # Loose cluster: a coarse direction + significant noise per item
    base_loose = rng.standard_normal(dim).astype(np.float32) * 0.3
    loose = np.stack(
        [base_loose + rng.standard_normal(dim).astype(np.float32) * 1.0 for _ in range(4)],
        axis=0,
    )

    # Normalize for compactness check (compactness expects L2-normalized).
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
        return m / norms

    tight_n = _normalize(tight)
    loose_n = _normalize(loose)
    c_tight = cluster_compactness(tight_n)
    c_loose = cluster_compactness(loose_n)
    print(f"[smoke] cluster compactness — tight: {c_tight:.3f}, loose: {c_loose:.3f}")
    assert c_tight > 0.95, f"tight cluster should be near-1, got {c_tight}"
    assert c_loose < c_tight, "loose cluster should be less compact than tight"

    base_threshold = 0.85
    t_tight = adaptive_threshold(base_threshold, tight_n, lambda_=0.5)
    t_loose = adaptive_threshold(base_threshold, loose_n, lambda_=0.5)
    print(f"[smoke] adaptive thresholds  — tight: {t_tight:.3f}, loose: {t_loose:.3f}")
    assert t_tight < t_loose, (
        "loose cluster should get a HIGHER (stricter) threshold so we keep its diversity"
    )
    assert abs(t_tight - base_threshold) < 1e-6, (
        "tight cluster threshold should be ~base (no need to tighten)"
    )

    # End-to-end: 8 pairs split 4 tight / 4 loose, two modalities.
    img = np.concatenate([tight, loose], axis=0)
    txt = np.concatenate([tight, loose], axis=0)  # same shape both modalities for simplicity
    pair_ids = [f"t{i+1:02d}" for i in range(4)] + [f"l{i+1:02d}" for i in range(4)]

    r_static = joint_similarity_dedup(
        pair_ids,
        modality_embeddings={"image": img, "text": txt},
        modality_weights={"image": 0.5, "text": 0.5},
        eps=0.05,
        alpha=0.7,
        n_clusters=2,
        adaptive_eps_lambda=0.0,  # legacy
    )
    r_adapt = joint_similarity_dedup(
        pair_ids,
        modality_embeddings={"image": img, "text": txt},
        modality_weights={"image": 0.5, "text": 0.5},
        eps=0.05,
        alpha=0.7,
        n_clusters=2,
        adaptive_eps_lambda=0.5,  # adaptive
    )

    print(
        f"[smoke] static drops: {r_static.stats['joint_drops']}, "
        f"adaptive drops: {r_adapt.stats['joint_drops']}"
    )
    print(f"[smoke] adaptive per-cluster thresholds: {r_adapt.stats['per_cluster_thresholds']}")

    pct = r_adapt.stats["per_cluster_thresholds"]
    assert pct, "adaptive run should report per-cluster thresholds"
    # The two cluster thresholds must differ (loose vs tight).
    vals = list(pct.values())
    assert max(vals) - min(vals) > 1e-3, (
        f"adaptive thresholds should diverge between clusters; got {vals}"
    )

    print("\n[smoke] adaptive_eps PASSED — per-cluster threshold gradient verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
