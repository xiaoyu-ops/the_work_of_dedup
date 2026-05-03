"""Text instantiation of the unified Q-SemDeDup framework.

Embedding: SBERT (sentence-transformers).
Quality:   Shannon entropy of the normalized text (alt: ``length``).
Selection: shared :mod:`pipelines.qsemdedup_core` (formula
           ``Score = alpha*Sim(x,C) + (1-alpha)*Norm(Quality(x))``).

Two execution modes:

- **single-stage** (default): SBERT-encode all texts, MiniBatchKMeans cluster,
  Q-SemDeDup within each cluster.
- **two-stage** (``config.two_stage=True``): MinHash LSH bucket candidates by
  char n-gram MinHash, then SBERT-encode only non-singleton buckets and run
  Q-SemDeDup per bucket. Recommended for >10k samples.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

from pipelines.qsemdedup_core import (
    kmeans_groups,
    lsh_buckets,
    normalize_quality,
    select_q_semdedup,
)


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


def _encode_texts(texts: Sequence[str], config: QSemDedupConfig) -> "np.ndarray":
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


def _text_signature(config: QSemDedupConfig):
    """Return a closure suitable for ``lsh_buckets``' signature_fn."""
    n_gram = max(1, config.lsh_ngram_size)
    cap = config.lsh_max_char_grams

    def _sig(text: str):
        cleaned = text.replace(" ", "")
        if len(cleaned) < n_gram:
            if cleaned:
                yield cleaned.encode("utf-8")
            return
        emitted = 0
        for i in range(len(cleaned) - n_gram + 1):
            if emitted >= cap:
                break
            yield cleaned[i : i + n_gram].encode("utf-8")
            emitted += 1

    return _sig


def _emit_dup_record(
    paths: Sequence[Path],
    keeper_global: int,
    dup_list: List[Any],
    threshold: float,
    cluster_id: int,
    stage: str,
) -> Dict[str, object]:
    return {
        "original": str(paths[keeper_global]),
        "duplicates": [
            {"path": str(paths[d_idx]), "similarity": float(sim)}
            for d_idx, sim in dup_list
        ],
        "similarity_threshold": float(threshold),
        "cluster_id": int(cluster_id),
        "stage": stage,
    }


def deduplicate_qsemdedup(
    paths: Sequence[Path],
    texts: Sequence[str],
    config: QSemDedupConfig,
) -> Dict[str, Any]:
    """Run text Q-SemDeDup on the given (path, normalized-text) pairs.

    Output mirrors the schema produced by other text dedup methods:
    ``keepers``, ``duplicates``, ``duplicate_count``, ``skipped``.
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
    qualities_norm = normalize_quality(qualities)
    threshold = 1.0 - float(config.eps)

    keep_flags = np.zeros(n, dtype=bool)
    duplicates_out: List[Dict[str, object]] = []
    duplicate_count = 0

    if config.two_stage:
        buckets = lsh_buckets(
            texts,
            _text_signature(config),
            threshold=config.lsh_threshold,
            num_perm=config.lsh_num_perm,
        )
        nontrivial = [b for b in buckets if len(b) >= max(2, config.min_cluster_size)]
        # Singletons (or sub-min-cluster buckets) are kept verbatim, no embedding.
        nontrivial_set = {id(b) for b in nontrivial}
        for bucket in buckets:
            if id(bucket) in nontrivial_set:
                continue
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
                result = select_q_semdedup(
                    bucket, member_feats, member_qual, threshold, config.alpha
                )
                for k in result["keep_indices"]:
                    keep_flags[k] = True
                for keeper_global, dup_list in result["dup_groups"].items():
                    duplicates_out.append(
                        _emit_dup_record(
                            paths, keeper_global, dup_list, threshold, bucket_id, "lsh+sbert"
                        )
                    )
                    duplicate_count += len(dup_list)
    else:
        embeddings = _encode_texts(texts, config)
        labels = kmeans_groups(
            embeddings,
            n_clusters=config.n_clusters,
            random_state=config.random_state,
            max_iter=config.kmeans_max_iter,
        )
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
            result = select_q_semdedup(
                members, member_feats, member_qual, threshold, config.alpha
            )
            for k in result["keep_indices"]:
                keep_flags[k] = True
            for keeper_global, dup_list in result["dup_groups"].items():
                duplicates_out.append(
                    _emit_dup_record(
                        paths, keeper_global, dup_list, threshold, cluster_id, "kmeans+sbert"
                    )
                )
                duplicate_count += len(dup_list)

    keepers = [paths[i] for i in range(n) if keep_flags[i]]
    return {
        "keepers": keepers,
        "duplicates": duplicates_out,
        "duplicate_count": duplicate_count,
        "skipped": 0,
    }
