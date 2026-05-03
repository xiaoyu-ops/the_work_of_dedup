"""Smoke test for text Q-SemDeDup on ~30 hand-crafted samples.

Run from repo root:
    uv run --extra text python experiments/scripts/smoke_qsemdedup_text.py

The script writes .txt files into a temp dir, runs the qsemdedup pipeline,
and prints per-cluster duplicate groups + final keepers/duplicates.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running as a plain script (`python experiments/scripts/...`) without
# installing the package — add repo root to sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from text.method.pipeline_api import (  # noqa: E402
    TextDedupConfig,
    TextEmbeddingConfig,
    TextPipelineConfig,
    run_text_pipeline,
)


# 3 semantic clusters, each with paraphrases + a couple of distinct outliers.
SAMPLES: list[tuple[str, str]] = [
    # cluster A — cats
    ("cat_01.txt", "The cat is sleeping on the warm sofa."),
    ("cat_02.txt", "A cat sleeps peacefully on the cozy couch."),
    ("cat_03.txt", "On the sofa, the cat is taking a nap."),
    ("cat_04.txt", "My cat curled up and fell asleep on the couch."),
    ("cat_05.txt", "The kitten is dozing off on the sofa."),
    ("cat_06.txt", "A small cat naps on the living room sofa."),
    ("cat_07.txt", "The feline rests quietly on the couch."),
    # cluster B — football
    ("foo_01.txt", "Manchester United won the football match last night."),
    ("foo_02.txt", "Man Utd defeated their rivals in yesterday's football game."),
    ("foo_03.txt", "Manchester United secured a victory in the soccer game."),
    ("foo_04.txt", "United beat the opposing team in last evening's match."),
    ("foo_05.txt", "The Red Devils triumphed in last night's football fixture."),
    ("foo_06.txt", "Manchester United claimed a win in their recent football tie."),
    # cluster C — weather
    ("wea_01.txt", "It is raining heavily in Beijing today."),
    ("wea_02.txt", "Beijing is experiencing a heavy rainfall right now."),
    ("wea_03.txt", "Heavy rain is falling on Beijing this afternoon."),
    ("wea_04.txt", "There is a downpour over Beijing at the moment."),
    ("wea_05.txt", "Beijing has heavy showers throughout the day."),
    # exact duplicates
    ("dup_01.txt", "The cat is sleeping on the warm sofa."),  # == cat_01
    ("dup_02.txt", "It is raining heavily in Beijing today."),  # == wea_01
    # distinct outliers (should all be kept)
    ("out_01.txt", "Quantum computing leverages superposition and entanglement."),
    ("out_02.txt", "The recipe calls for two cups of flour and one egg."),
    ("out_03.txt", "She finished reading the philosophy textbook in three days."),
    ("out_04.txt", "The mountain trail was steep but the view was rewarding."),
    ("out_05.txt", "He installed a new SSD to speed up his old laptop."),
    ("out_06.txt", "Annual GDP growth slowed to 3.1% last quarter."),
    ("out_07.txt", "The orchestra performed Beethoven's ninth symphony."),
    ("out_08.txt", "Mars rovers have collected samples for years."),
    ("out_09.txt", "Bees pollinate a third of the food crops we eat."),
    ("out_10.txt", "Linguistics studies the structure and evolution of language."),
]


def _run_mode(paths, label: str, cfg: TextPipelineConfig) -> set[str]:
    print(f"\n========== mode: {label} ==========")
    result = run_text_pipeline(paths, cfg)

    print("[smoke] stats:")
    for k, v in result.stats.items():
        print(f"  {k}: {v}")

    print(f"\n[smoke] keepers ({len(result.keepers)}):")
    for k in result.keepers:
        print(f"  + {k.name}")

    print(f"\n[smoke] duplicate groups ({len(result.duplicates)}):")
    for grp in result.duplicates:
        orig = Path(grp["original"]).name
        cluster = grp.get("cluster_id", "?")
        stage = grp.get("stage", "?")
        print(f"  - keeper={orig}  cluster={cluster}  stage={stage}")
        for d in grp["duplicates"]:
            dpath = Path(d["path"]).name
            print(f"      ~ {dpath}  sim={d['similarity']:.3f}")

    keeper_names = {p.name for p in result.keepers}
    assert not ({"cat_01.txt", "dup_01.txt"}.issubset(keeper_names)), (
        f"[{label}] exact dup_01.txt should collapse with cat_01.txt"
    )
    assert not ({"wea_01.txt", "dup_02.txt"}.issubset(keeper_names)), (
        f"[{label}] exact dup_02.txt should collapse with wea_01.txt"
    )
    for i in range(1, 11):
        name = f"out_{i:02d}.txt"
        assert name in keeper_names, f"[{label}] distinct outlier {name} dropped"

    print(f"[smoke:{label}] OK — exact dups collapsed, all distinct outliers kept.")
    return keeper_names


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qsem_smoke_") as tmp:
        tmp_dir = Path(tmp)
        paths: list[Path] = []
        for name, body in SAMPLES:
            p = tmp_dir / name
            p.write_text(body, encoding="utf-8")
            paths.append(p)

        print(f"[smoke] wrote {len(paths)} sample files to {tmp_dir}")

        # Mode 1: single-stage (KMeans + SBERT)
        single_cfg = TextPipelineConfig(
            embedding=TextEmbeddingConfig(),
            dedup=TextDedupConfig(
                method="qsemdedup",
                n_clusters=3,
                eps=0.15,
                alpha=0.7,
                quality_metric="entropy",
                sbert_device="cpu",
                two_stage=False,
            ),
        )
        _run_mode(paths, "single-stage", single_cfg)

        # Mode 2: two-stage (MinHash LSH coarse filter + SBERT precision)
        two_stage_cfg = TextPipelineConfig(
            embedding=TextEmbeddingConfig(),
            dedup=TextDedupConfig(
                method="qsemdedup",
                eps=0.15,
                alpha=0.7,
                quality_metric="entropy",
                sbert_device="cpu",
                two_stage=True,
                lsh_threshold=0.5,
                lsh_num_perm=128,
                lsh_ngram_size=5,
            ),
        )
        _run_mode(paths, "two-stage", two_stage_cfg)

        print("\n[smoke] both modes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
