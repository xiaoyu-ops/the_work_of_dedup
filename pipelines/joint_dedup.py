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
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set


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
