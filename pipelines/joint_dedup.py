"""Cross-modality joint deduplication for the unified Q-SemDeDup framework.

The per-modality runners (image / text / audio) each produce a ``keepers`` list
that is *modality-local* — they do not know about cross-modal pairing. For a
multimodal corpus (e.g., CC3M image-caption pairs, AudioCaps audio-caption
pairs) the natural ground truth is the **pair**, not the individual file. Plan
A node 2 specifies the rule:

    "同一图文对中，图像或文本任一被去重，则整个对丢弃"

This module turns that rule into a small, testable function. The rest of the
pipeline (orchestrator, runners, sorter) does not need to know about pairing —
this stage runs *after* the modality stages, reads their results, and produces
a joint result.

Pairing is driven by a caller-supplied function ``pair_id_fn(Path) -> str``.
The default uses the file stem, which is correct for datasets that name
parallel modalities with the same base name (``001234.jpg`` ↔ ``001234.txt``).
For non-trivial layouts the caller can pass a custom function.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


PairIdFn = Callable[[Path], str]


def stem_pair_id(path: Path) -> str:
    """Default pairing strategy: file stem (basename without extension).

    Suitable for datasets like CC3M / LAION subsets where each (image, caption)
    pair is named ``<id>.jpg`` and ``<id>.txt``. For audio-text pairs the same
    rule applies (``<id>.wav`` and ``<id>.txt``).
    """
    return Path(path).stem


@dataclass
class ModalityKeepers:
    """Keepers reported by a single modality runner."""

    name: str
    keepers: Sequence[Path]


@dataclass
class JointDedupResult:
    """Outcome of intersecting per-modality keepers by pair id."""

    pair_keepers: Set[str] = field(default_factory=set)
    """Pair ids that survived in *every* modality — the joint keep set."""

    per_modality_pairs: Dict[str, Set[str]] = field(default_factory=dict)
    """Pair ids each modality kept (modality name -> pair id set)."""

    pair_drops: Dict[str, List[str]] = field(default_factory=dict)
    """For dropped pairs, the modalities that dropped them
    (pair_id -> list of modality names that did NOT keep it)."""

    stats: Dict[str, int] = field(default_factory=dict)
    """Aggregate counts (input_modalities, joint_keepers, joint_drops, ...)."""


def join_keepers(
    modality_results: Sequence[ModalityKeepers],
    pair_id_fn: PairIdFn = stem_pair_id,
) -> JointDedupResult:
    """Intersect keepers across modalities by pair id.

    A pair survives only if *every* modality kept it. Pairs present in some
    modalities but not others are dropped, and the list of modalities that
    dropped each pair is recorded in ``pair_drops`` for downstream analysis
    (per-modality dedup-rate breakdown, error attribution, etc.).
    """
    result = JointDedupResult()
    if not modality_results:
        result.stats = {"input_modalities": 0}
        return result

    per_modality_pairs: Dict[str, Set[str]] = {}
    all_pair_ids: Set[str] = set()
    for mod in modality_results:
        ids = {pair_id_fn(Path(p)) for p in mod.keepers}
        per_modality_pairs[mod.name] = ids
        all_pair_ids |= ids

    pair_keepers: Set[str] = set.intersection(*per_modality_pairs.values()) if per_modality_pairs else set()

    pair_drops: Dict[str, List[str]] = {}
    for pid in all_pair_ids - pair_keepers:
        dropped_by = [name for name, ids in per_modality_pairs.items() if pid not in ids]
        pair_drops[pid] = dropped_by

    stats: Dict[str, int] = {
        "input_modalities": len(modality_results),
        "joint_keepers": len(pair_keepers),
        "joint_drops": len(pair_drops),
        "total_unique_pairs": len(all_pair_ids),
    }
    for mod in modality_results:
        stats[f"{mod.name}_keepers"] = len(per_modality_pairs[mod.name])

    result.pair_keepers = pair_keepers
    result.per_modality_pairs = per_modality_pairs
    result.pair_drops = pair_drops
    result.stats = stats
    return result


# ---------------------------------------------------------------------------
# Loaders for orchestrator-emitted artifacts
# ---------------------------------------------------------------------------

def load_keepers_from_output_dir(
    output_dir: Path,
    extensions: Optional[Iterable[str]] = None,
) -> List[Path]:
    """List kept files copied by a modality runner into its output directory.

    The runners use :func:`pipelines.modalities.common.copy_existing_files` to
    materialize keepers in ``output_dir``. This helper lists those files so
    callers can build a :class:`ModalityKeepers` without re-running anything.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    out: List[Path] = []
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or [])}
    for child in output_dir.rglob("*"):
        if not child.is_file():
            continue
        if exts and child.suffix.lower() not in exts:
            continue
        # Skip the orchestrator's own artifact files.
        if child.name.endswith("_runner_summary.json") or child.name.endswith("_duplicates.json"):
            continue
        out.append(child)
    return out


def load_keepers_from_summary(summary_path: Path) -> List[Path]:
    """Recover keepers from a runner summary JSON, if it embeds them.

    Older runners do not embed keepers in the summary (they only write counts);
    callers should fall back to :func:`load_keepers_from_output_dir`.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return []
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("keepers") or []
    return [Path(p) for p in raw]


# ---------------------------------------------------------------------------
# Direction B: joint-similarity dedup (replaces "AND-of-per-modality")
# ---------------------------------------------------------------------------
#
# Plan A innovation #2 — Direction B.
#
# Existing pipelines (SemDeDup / D4 / FairDeDup / EcoDatum) treat modalities
# independently: dedup image, dedup text, then take the intersection. That
# rule is OR-shaped — a pair is dropped if EITHER modality flagged it. This
# discards pairs that are near-duplicate in one modality but contribute
# diversity in the other (e.g., same image with paraphrased captions —
# valuable for MLLM instruction tuning).
#
# Direction B replaces the OR rule with a weighted JOINT similarity:
#
#     sim_joint(pair_i, pair_j) = Σ_m  w_m · cos( emb_m(i), emb_m(j) )
#
# Two pairs collapse only when their joint similarity exceeds the threshold,
# i.e., they are near-duplicate in *all* modalities simultaneously. The
# decision moves from per-modality dedup to a true multimodal dedup decision.

@dataclass
class JointSimilarityResult:
    """Return shape for :func:`joint_similarity_dedup`.

    ``pair_keepers`` is the surviving set; ``dup_groups`` maps each keeper
    pair_id to the list of (dropped_pair_id, sim_joint) it absorbed.
    """

    pair_keepers: List[str] = field(default_factory=list)
    dup_groups: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


def _l2_normalize(matrix: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return (matrix / norms).astype(np.float32, copy=False)


def _build_joint_embedding(
    modality_embeddings: Dict[str, "np.ndarray"],
    modality_weights: Dict[str, float],
) -> "np.ndarray":
    """Weighted concatenation of L2-normalized per-modality embeddings.

    Modalities present in ``modality_weights`` but missing from
    ``modality_embeddings`` are skipped. Each block is L2-normalized and
    scaled by ``sqrt(w_m)`` so that the dot product of two joint embeddings
    equals ``Σ w_m · cos(emb_m(i), emb_m(j))``.
    """
    if np is None:
        raise RuntimeError("numpy is required for joint_similarity_dedup")
    aligned: List["np.ndarray"] = []
    weight_sum = sum(modality_weights.get(m, 0.0) for m in modality_embeddings.keys())
    if weight_sum <= 0:
        raise ValueError("modality_weights must contain positive entries")

    n_rows: Optional[int] = None
    for modality, emb in modality_embeddings.items():
        w = modality_weights.get(modality, 0.0)
        if w <= 0:
            continue
        if emb is None or emb.size == 0:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if n_rows is None:
            n_rows = emb.shape[0]
        elif emb.shape[0] != n_rows:
            raise ValueError(
                f"all modality embeddings must share row count; "
                f"got {emb.shape[0]} for {modality} (expected {n_rows})"
            )
        block = _l2_normalize(emb) * float(np.sqrt(w / weight_sum))
        aligned.append(block.astype(np.float32, copy=False))

    if not aligned or n_rows is None:
        raise ValueError("no non-empty modality embeddings provided")

    joint = np.concatenate(aligned, axis=1)
    # Re-normalize the joint embedding so cosine on it is bounded in [-1, 1].
    return _l2_normalize(joint)


def joint_similarity_dedup(
    pair_ids: Sequence[str],
    modality_embeddings: Dict[str, "np.ndarray"],
    modality_weights: Dict[str, float],
    *,
    quality: Optional["np.ndarray"] = None,
    eps: float = 0.05,
    alpha: float = 0.7,
    n_clusters: Optional[int] = None,
    adaptive_eps_lambda: float = 0.0,
) -> JointSimilarityResult:
    """Pair-level Q-SemDeDup on a weighted joint embedding.

    Parameters
    ----------
    pair_ids
        N pair identifiers (e.g. file stems). Each row of every entry in
        ``modality_embeddings`` corresponds to ``pair_ids[i]``.
    modality_embeddings
        ``{modality_name: [N, D_m]}`` per-modality embedding matrices. Missing
        modalities for individual pairs should be represented as zero rows by
        the caller.
    modality_weights
        ``{modality_name: weight}`` non-negative weights. Re-normalized to
        sum to 1 internally.
    quality
        Optional ``[N]`` quality vector (e.g. cross-modal alignment from
        Direction A). When provided, drives in-cluster ranking via
        ``Score = alpha·Sim(x,C) + (1-alpha)·Norm(quality)``.
    eps
        ``threshold = 1 - eps`` over joint cosine similarity.
    alpha
        Q-SemDeDup quality / cohesion trade-off.
    n_clusters
        KMeans cluster count for grouping; ``None`` triggers the
        ``max(1, N // 100)`` heuristic shared with the per-modality runners.
    adaptive_eps_lambda
        Plan A direction C: per-cluster ε scaled by cluster compactness.
        ``0.0`` (default) keeps the constant ``eps`` everywhere — equivalent
        to the original Q-SemDeDup. Positive values tighten the threshold for
        diverse clusters (preserving cross-modal-rich content) and loosen it
        for tight clusters (collapsing redundancies aggressively).
    """
    if np is None:
        raise RuntimeError("numpy is required for joint_similarity_dedup")
    if not pair_ids:
        return JointSimilarityResult(stats={"input_pairs": 0})

    from pipelines.qsemdedup_core import (
        adaptive_threshold,
        kmeans_groups,
        normalize_quality,
        select_q_semdedup,
    )

    n = len(pair_ids)
    joint_emb = _build_joint_embedding(modality_embeddings, modality_weights)
    if joint_emb.shape[0] != n:
        raise ValueError(
            f"pair_ids length {n} does not match joint embedding rows "
            f"{joint_emb.shape[0]}"
        )

    if quality is None:
        quality = np.zeros(n, dtype=np.float32)
    quality_norm = normalize_quality(np.asarray(quality, dtype=np.float32))

    threshold = 1.0 - float(eps)
    labels = kmeans_groups(joint_emb, n_clusters=n_clusters)
    cluster_to_idx: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        cluster_to_idx.setdefault(int(lab), []).append(idx)

    keep_flags = np.zeros(n, dtype=bool)
    dup_groups: Dict[str, List[Tuple[str, float]]] = {}
    duplicate_count = 0

    per_cluster_thresholds: Dict[int, float] = {}
    for cluster_id, members in cluster_to_idx.items():
        if len(members) < 2:
            for idx in members:
                keep_flags[idx] = True
            continue
        member_feats = joint_emb[members]
        member_qual = quality_norm[members]
        cluster_threshold = (
            adaptive_threshold(threshold, member_feats, lambda_=adaptive_eps_lambda)
            if adaptive_eps_lambda > 0
            else threshold
        )
        per_cluster_thresholds[int(cluster_id)] = cluster_threshold
        result = select_q_semdedup(
            members, member_feats, member_qual, cluster_threshold, alpha
        )
        for k in result["keep_indices"]:
            keep_flags[k] = True
        for keeper_global, dups in result["dup_groups"].items():
            keeper_id = pair_ids[keeper_global]
            entries = dup_groups.setdefault(keeper_id, [])
            for d_idx, sim in dups:
                entries.append((pair_ids[d_idx], float(sim)))
                duplicate_count += 1

    keepers = [pair_ids[i] for i in range(n) if keep_flags[i]]
    return JointSimilarityResult(
        pair_keepers=keepers,
        dup_groups=dup_groups,
        stats={
            "input_pairs": n,
            "joint_keepers": len(keepers),
            "joint_drops": duplicate_count,
            "n_clusters_actual": len(cluster_to_idx),
            "threshold": threshold,
            "alpha": alpha,
            "adaptive_eps_lambda": adaptive_eps_lambda,
            "per_cluster_thresholds": per_cluster_thresholds,
            "modality_weights": dict(modality_weights),
        },
    )


def write_joint_summary(result: JointDedupResult, summary_path: Path) -> None:
    """Persist a :class:`JointDedupResult` as JSON for the report stage."""
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pair_keepers": sorted(result.pair_keepers),
        "per_modality_pairs": {k: sorted(v) for k, v in result.per_modality_pairs.items()},
        "pair_drops": result.pair_drops,
        "stats": result.stats,
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
