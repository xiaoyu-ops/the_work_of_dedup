"""Unified Quality-Aware Multimodal Deduplication core (Q-SemDeDup).

This module is the cross-modal heart of the framework described in the paper's
Section 3 ("Unified Quality-Aware Multimodal Deduplication Framework"). The
image, text, and audio runners each compute their own embeddings and quality
signals, but they all funnel through the same selection rule:

    Score(x) = alpha * Sim(x, C) + (1 - alpha) * Norm(Quality(x))         [*]

where
- ``C`` is the centroid of the candidate group the item belongs to,
- ``Sim`` is cosine similarity between the L2-normalized embedding ``x`` and
  ``C`` (so it lies in [-1, 1] and is in [0, 1] for sensible groups),
- ``Quality(x)`` is a modality-specific signal (image: file size / resolution;
  text: Shannon entropy / length; audio: SNR / effective duration), normalized
  to [0, 1] over the global population of the run.

The candidate-group abstraction is intentionally generic. Two grouping schemes
are available out of the box:

- :func:`kmeans_groups` — MiniBatchKMeans clustering of dense embeddings;
  used when every item has an embedding (small / mid scale).
- :func:`lsh_buckets` — MinHash LSH coarse buckets keyed by a caller-supplied
  signature builder; used as the first stage of two-stage pipelines so that
  expensive deep encoders only run on bucketed candidates.

After grouping, callers invoke :func:`select_q_semdedup` once per group to
greedily keep the highest-scoring item and mark the rest as duplicates.

Heavy dependencies (``numpy``, ``scikit-learn``, ``datasketch``) are imported
lazily so importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a base dependency
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Quality normalization
# ---------------------------------------------------------------------------

def normalize_quality(values: "np.ndarray") -> "np.ndarray":
    """Min-max normalize a quality vector to ``[0, 1]``.

    A constant vector is collapsed to zeros so it contributes nothing to the
    Q-SemDeDup score (selection then falls back to pure similarity).
    """
    if np is None:
        raise RuntimeError("numpy is required for normalize_quality")
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Selection (formula [*])
# ---------------------------------------------------------------------------

SelectResult = Dict[str, Any]


def cluster_compactness(member_feats: "np.ndarray") -> float:
    """Mean cosine similarity to the centroid for an L2-normalized cluster.

    Returns a value in ``[0, 1]`` for unit-norm rows. Tight clusters (true
    near-duplicates) approach 1; loose clusters (diverse concept) approach 0.
    Used by :func:`adaptive_threshold` to make ε cluster-specific so that
    diverse clusters are pruned conservatively.
    """
    if np is None:
        raise RuntimeError("numpy is required for cluster_compactness")
    if member_feats.shape[0] == 0:
        return 0.0
    centroid = member_feats.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    sims = member_feats @ centroid
    return float(np.clip(sims.mean(), 0.0, 1.0))


def adaptive_threshold(
    base_threshold: float,
    member_feats: "np.ndarray",
    *,
    lambda_: float = 0.5,
) -> float:
    """Per-cluster threshold tightened by cluster compactness.

    Plan A direction C — extends DBP-style adaptive pruning to the multimodal
    setting. Tight clusters (truly redundant content) get a *lower* threshold
    so more items are flagged as duplicates; loose clusters (diverse concept)
    get a *higher* threshold so we keep more diversity:

        threshold_cluster = base_threshold - λ · (1 - compactness)
                                                 · (1 - base_threshold)

    Equivalently in eps-space::

        eps_cluster = base_eps · (1 + λ · (1 - compactness))    [larger eps
                                                                 for loose
                                                                 clusters]

    Wait — that is the opposite of intent. Re-derived more carefully::

        compactness ≈ 1  → cluster is uniform  → loosen threshold (keep base
                                                  or below) so we still drop
                                                  redundancies aggressively.
        compactness ≈ 0  → cluster is diverse  → tighten threshold so only
                                                  near-identical items collapse.

    So tight cluster ⇒ lower threshold ⇒ more drops. Loose cluster ⇒ higher
    threshold ⇒ fewer drops. The closed form below implements that.
    ``lambda_=0`` recovers the constant base threshold.
    """
    if np is None:
        raise RuntimeError("numpy is required for adaptive_threshold")
    if member_feats.shape[0] < 2:
        return float(base_threshold)
    base = float(base_threshold)
    diversity = 1.0 - cluster_compactness(member_feats)
    # Headroom toward the strict end (1.0) to avoid pushing threshold above 1.
    headroom = 1.0 - base
    delta = float(lambda_) * diversity * headroom
    return float(np.clip(base + delta, 0.0, 1.0))


def select_q_semdedup(
    member_global_idx: Sequence[int],
    member_feats: "np.ndarray",
    member_qual_norm: "np.ndarray",
    threshold: float,
    alpha: float = 0.7,
) -> SelectResult:
    """Greedy Q-SemDeDup selection inside a candidate group.

    Parameters
    ----------
    member_global_idx
        Global indices (into the run-level item list) of the M members in this
        group.
    member_feats
        L2-normalized embedding matrix of shape ``[M, D]`` for the same M items.
    member_qual_norm
        Pre-normalized quality scores of shape ``[M]`` (already in ``[0, 1]``).
    threshold
        Cosine similarity threshold; items whose nearest-keeper similarity is
        ``>= threshold`` are marked as duplicates. Typically ``threshold = 1 - eps``.
    alpha
        Trade-off between cluster cohesion (``alpha`` term, "stay close to the
        centroid") and quality (``1 - alpha`` term, "prefer high-quality items").
        ``alpha = 1`` recovers SemDeDup; ``alpha = 0`` selects purely by quality.

    Returns
    -------
    dict with two keys:
        ``keep_indices`` : list of global indices that survived selection.
        ``dup_groups``   : ``{keeper_global_idx: [(dup_global_idx, sim), ...]}``
                            mapping each keeper to the duplicates it absorbed.
    """
    if np is None:
        raise RuntimeError("numpy is required for select_q_semdedup")
    if len(member_global_idx) == 0:
        return {"keep_indices": [], "dup_groups": {}}
    if len(member_global_idx) == 1:
        return {"keep_indices": list(member_global_idx), "dup_groups": {}}

    # Centroid-based cohesion: cosine sim of each item to the group centroid.
    centroid = member_feats.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    sim_to_center = member_feats @ centroid

    score = alpha * sim_to_center + (1.0 - alpha) * member_qual_norm
    order = np.argsort(-score)  # descending: best score first

    kept_local: List[int] = []
    kept_feats: List["np.ndarray"] = []
    dup_groups: Dict[int, List[Tuple[int, float]]] = {}

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
            keeper_global = int(member_global_idx[best_keeper_local])
            dup_groups.setdefault(keeper_global, []).append(
                (int(member_global_idx[int(local_i)]), best_sim)
            )
        else:
            kept_local.append(int(local_i))
            kept_feats.append(cur)

    keep_indices = [int(member_global_idx[i]) for i in kept_local]
    return {"keep_indices": keep_indices, "dup_groups": dup_groups}


# ---------------------------------------------------------------------------
# Grouping strategy 1: dense KMeans
# ---------------------------------------------------------------------------

def auto_n_clusters(n_items: int, items_per_cluster: int = 100, cap: int = 50_000) -> int:
    """Pick a reasonable cluster count when the caller does not specify one."""
    if n_items <= 1:
        return 1
    return max(1, min(n_items // max(1, items_per_cluster), cap))


def kmeans_groups(
    embeddings: "np.ndarray",
    n_clusters: Optional[int] = None,
    *,
    random_state: int = 42,
    max_iter: int = 100,
    items_per_cluster: int = 100,
) -> "np.ndarray":
    """Group dense embeddings into clusters with MiniBatchKMeans.

    Returns an ``int32`` array of cluster labels. ``n_clusters=None`` triggers
    the same heuristic as the image pipeline: ``max(1, N / 100)``.
    """
    if np is None:
        raise RuntimeError("numpy is required for kmeans_groups")
    n_items = int(embeddings.shape[0])
    k = n_clusters if (n_clusters is not None and n_clusters > 0) else auto_n_clusters(
        n_items, items_per_cluster=items_per_cluster
    )
    k = max(1, min(k, n_items))
    if k <= 1:
        return np.zeros(n_items, dtype=np.int32)
    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for kmeans_groups; install via "
            "`uv sync --extra image` or `--extra text`"
        ) from exc
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(1024, max(32, n_items)),
        max_iter=max_iter,
        n_init="auto",
    )
    return kmeans.fit_predict(embeddings).astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# Grouping strategy 2: MinHash LSH coarse buckets (two-stage pipelines)
# ---------------------------------------------------------------------------

SignatureFn = Callable[[Any], Iterable[bytes]]
"""A function ``item -> iterable of byte-strings`` to be hashed into MinHash."""


def lsh_buckets(
    items: Sequence[Any],
    signature_fn: SignatureFn,
    *,
    threshold: float = 0.5,
    num_perm: int = 128,
) -> List[List[int]]:
    """Coarse-bucket a list of items by MinHash LSH similarity.

    The caller passes ``signature_fn`` so each modality decides what to hash
    (text: char n-grams; audio: spectrogram peak hashes; image: pHash bits).

    Returns a list of buckets, each a list of global indices. Items that match
    nothing land in their own singleton bucket. Transitive matches are merged
    via union-find so a bucket reflects a connected component of LSH hits.
    """
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "datasketch is required for lsh_buckets; install via `uv sync --extra text`"
        ) from exc

    n = len(items)
    if n == 0:
        return []

    minhashes: List[Any] = []
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, item in enumerate(items):
        m = MinHash(num_perm=num_perm)
        for token in signature_fn(item):
            m.update(token)
        minhashes.append(m)
        lsh.insert(f"item_{i}", m)

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
