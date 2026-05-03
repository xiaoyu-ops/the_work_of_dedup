"""Text Q-SemDeDup: SBERT embedding + clustering + entropy-aware selection.

Implements the text-modality instantiation of the unified Q-SemDeDup framework:

    Score(x) = alpha * Sim(x, C) + (1 - alpha) * Norm(Quality(x))

where C is the cluster centroid, Sim is cosine similarity, and Quality is a
text-level signal (default: Shannon entropy of the normalized text).

Heavy dependencies (sentence-transformers, scikit-learn, numpy) are imported
lazily so importing this module is cheap and does not break the base install.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class QSemDedupConfig:
    """Runtime knobs for text Q-SemDeDup."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"  # auto | cpu | cuda
    batch_size: int = 64
    n_clusters: Optional[int] = None  # None => auto: max(1, len // 100)
    min_cluster_size: int = 2
    eps: float = 0.05  # cosine-distance threshold; threshold = 1 - eps
    alpha: float = 0.7
    quality_metric: str = "entropy"  # entropy | length
    random_state: int = 42
    kmeans_max_iter: int = 100

    # --- Two-stage (MinHash LSH coarse filter -> SBERT precision) ---
    # When enabled, candidates are first grouped by MinHash LSH and only
    # non-singleton buckets are SBERT-encoded. This is the recommended path
    # for >10k samples and the configuration described in plan A.
    two_stage: bool = False
    lsh_threshold: float = 0.5
    lsh_num_perm: int = 128
    lsh_ngram_size: int = 5
    lsh_max_char_grams: int = 200


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _quality_score(text: str, metric: str) -> float:
    if metric == "length":
        return float(len(text))
    return _shannon_entropy(text)


def _encode_texts(
    texts: Sequence[str],
    config: QSemDedupConfig,
) -> "np.ndarray":
    """Run SBERT to produce L2-normalized embeddings."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required for qsemdedup; install via "
            "`uv sync --extra text`"
        ) from exc

    device = _resolve_device(config.device)
    model = SentenceTransformer(config.model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=config.batch_size,
        show_progress_bar=len(texts) > 256,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32, copy=False)


def _decide_n_clusters(n_items: int, requested: Optional[int]) -> int:
    if requested is not None and requested > 0:
        return min(requested, max(1, n_items))
    # Heuristic: ~100 items per cluster, cap at 50k, floor at 1.
    return max(1, min(n_items // 100, 50000))


def _cluster(embeddings: "np.ndarray", config: QSemDedupConfig) -> "np.ndarray":
    n_items = embeddings.shape[0]
    n_clusters = _decide_n_clusters(n_items, config.n_clusters)
    if n_clusters <= 1:
        return np.zeros(n_items, dtype=np.int32)
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for qsemdedup clustering"
        ) from exc

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        batch_size=min(1024, max(32, n_items)),
        max_iter=config.kmeans_max_iter,
        n_init="auto",
    )
    return kmeans.fit_predict(embeddings).astype(np.int32, copy=False)


def _normalize_quality(values: "np.ndarray") -> "np.ndarray":
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _select_q_semdedup(
    member_global_idx: List[int],
    member_feats: "np.ndarray",
    member_qual_norm: "np.ndarray",
    threshold: float,
    alpha: float,
) -> Dict[str, Any]:
    """Greedy Q-SemDeDup selection over an arbitrary group.

    `member_feats` is the L2-normalized [M, D] feature matrix for the group;
    `member_qual_norm` is its [M] quality vector (already min-max normalized
    over the global population). Returns dict with `keep_indices` (global)
    and `dup_groups` mapping keeper_global_idx -> list of (dup_global_idx, sim).
    """
    if len(member_global_idx) == 1:
        return {"keep_indices": list(member_global_idx), "dup_groups": {}}

    centroid = member_feats.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    sim_to_center = member_feats @ centroid

    score = alpha * sim_to_center + (1.0 - alpha) * member_qual_norm
    order = np.argsort(-score)

    kept_local: List[int] = []
    kept_feats: List["np.ndarray"] = []
    dup_groups: Dict[int, List[Any]] = {}

    for local_i in order:
        cur = member_feats[local_i]
        is_dup = False
        best_keeper_local = -1
        best_sim = -1.0
        if kept_feats:
            arr = np.stack(kept_feats, axis=0)
            sims = arr @ cur
            j = int(np.argmax(sims))
            best_sim = float(sims[j])
            best_keeper_local = kept_local[j]
            if best_sim >= threshold:
                is_dup = True

        if is_dup:
            keeper_global = member_global_idx[best_keeper_local]
            dup_groups.setdefault(keeper_global, []).append(
                (member_global_idx[int(local_i)], best_sim)
            )
        else:
            kept_local.append(int(local_i))
            kept_feats.append(cur)

    keep_indices = [member_global_idx[i] for i in kept_local]
    return {"keep_indices": keep_indices, "dup_groups": dup_groups}


def _char_ngrams_no_space(text: str, n: int) -> List[str]:
    cleaned = text.replace(" ", "")
    if len(cleaned) < n:
        return [] if not cleaned else [cleaned]
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def _lsh_candidate_buckets(
    texts: Sequence[str],
    config: QSemDedupConfig,
) -> List[List[int]]:
    """Stage 1: bucket candidate duplicates with MinHash LSH.

    Returns a list of buckets (each is a list of global indices). Items not
    matching anything land in their own singleton bucket. We use a small
    union-find so transitively-linked items end up in the same bucket.
    """
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasketch is required for two-stage qsemdedup; install via "
            "`uv sync --extra text`"
        ) from exc

    n = len(texts)
    minhashes: List[Any] = []
    lsh = MinHashLSH(threshold=config.lsh_threshold, num_perm=config.lsh_num_perm)

    for i, text in enumerate(texts):
        m = MinHash(num_perm=config.lsh_num_perm)
        grams = _char_ngrams_no_space(text, max(1, config.lsh_ngram_size))
        for g in grams[: config.lsh_max_char_grams]:
            m.update(g.encode("utf-8"))
        minhashes.append(m)
        lsh.insert(f"doc_{i}", m)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for key in lsh.query(minhashes[i]):
            j = int(key.split("_", 1)[1])
            if j != i:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def deduplicate_qsemdedup(
    paths: Sequence[Path],
    texts: Sequence[str],
    config: QSemDedupConfig,
) -> Dict[str, Any]:
    """Run text Q-SemDeDup on the given (path, normalized-text) pairs.

    Two execution modes:
      - single-stage (default): SBERT-encode all texts, MiniBatchKMeans cluster,
        Q-SemDeDup within each cluster.
      - two-stage (``config.two_stage=True``): MinHash LSH bucket candidates
        first, then SBERT-encode only non-singleton buckets and run Q-SemDeDup
        per bucket. Recommended for >10k samples.

    Output mirrors the schema produced by other text dedup methods:
    `keepers`, `duplicates`, `duplicate_count`, `skipped`.
    """
    if np is None:
        raise RuntimeError("numpy is required for qsemdedup")
    if len(paths) != len(texts):
        raise ValueError("paths and texts must have the same length")
    n = len(paths)
    if n == 0:
        return {"keepers": [], "duplicates": [], "duplicate_count": 0, "skipped": 0}

    qualities = np.array(
        [_quality_score(t, config.quality_metric) for t in texts],
        dtype=np.float32,
    )
    qualities_norm = _normalize_quality(qualities)
    threshold = 1.0 - float(config.eps)

    keep_flags = np.zeros(n, dtype=bool)
    duplicates_out: List[Dict[str, object]] = []
    duplicate_count = 0

    if config.two_stage:
        buckets = _lsh_candidate_buckets(texts, config)
        nontrivial = [b for b in buckets if len(b) >= 2]
        # Singletons are kept verbatim, no embedding needed.
        for bucket in buckets:
            if len(bucket) < max(1, config.min_cluster_size):
                for idx in bucket:
                    keep_flags[idx] = True

        if nontrivial:
            embed_global_idx: List[int] = [i for b in nontrivial for i in b]
            embed_texts = [texts[i] for i in embed_global_idx]
            sub_embeddings = _encode_texts(embed_texts, config)
            global_to_sub = {gi: si for si, gi in enumerate(embed_global_idx)}

            for bucket_id, bucket in enumerate(nontrivial):
                local_rows = [global_to_sub[i] for i in bucket]
                member_feats = sub_embeddings[local_rows]
                member_qual = qualities_norm[bucket]
                result = _select_q_semdedup(
                    bucket, member_feats, member_qual, threshold, config.alpha
                )
                for k in result["keep_indices"]:
                    keep_flags[k] = True
                for keeper_global, dup_list in result["dup_groups"].items():
                    duplicates_out.append(
                        {
                            "original": str(paths[keeper_global]),
                            "duplicates": [
                                {"path": str(paths[d_idx]), "similarity": float(sim)}
                                for d_idx, sim in dup_list
                            ],
                            "similarity_threshold": float(threshold),
                            "cluster_id": int(bucket_id),
                            "stage": "lsh+sbert",
                        }
                    )
                    duplicate_count += len(dup_list)
    else:
        embeddings = _encode_texts(texts, config)
        labels = _cluster(embeddings, config)
        cluster_to_members: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            cluster_to_members.setdefault(int(lab), []).append(idx)

        for cluster_id, members in cluster_to_members.items():
            if len(members) < max(1, config.min_cluster_size):
                for idx in members:
                    keep_flags[idx] = True
                continue
            member_feats = embeddings[members]
            member_qual = qualities_norm[members]
            result = _select_q_semdedup(
                members, member_feats, member_qual, threshold, config.alpha
            )
            for k in result["keep_indices"]:
                keep_flags[k] = True
            for keeper_global, dup_list in result["dup_groups"].items():
                duplicates_out.append(
                    {
                        "original": str(paths[keeper_global]),
                        "duplicates": [
                            {"path": str(paths[d_idx]), "similarity": float(sim)}
                            for d_idx, sim in dup_list
                        ],
                        "similarity_threshold": float(threshold),
                        "cluster_id": int(cluster_id),
                        "stage": "kmeans+sbert",
                    }
                )
                duplicate_count += len(dup_list)

    keepers = [paths[i] for i in range(n) if keep_flags[i]]
    return {
        "keepers": keepers,
        "duplicates": duplicates_out,
        "duplicate_count": duplicate_count,
        "skipped": 0,
    }
