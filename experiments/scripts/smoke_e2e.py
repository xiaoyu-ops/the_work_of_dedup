"""End-to-end smoke for the full multimodal pipeline.

Generates a synthetic 10-pair mini corpus into a temp dir, writes a portable
pipeline config that uses lightweight backends (image: average_rgb; text: SBERT
all-MiniLM-L6-v2; audio: disabled), then drives the real
:func:`pipelines.multimodal_runner.main` and verifies:

- Stage sentinels exist:
    stage1_sorter/_SUCCESS, stage2_image/_SUCCESS, stage2_text/_SUCCESS,
    stage3_joint_dedup/_SUCCESS
- ``joint_dedup_summary.json`` exists with sane stats (1..10 keepers and a
  positive total_unique_pairs).

This is the "does the pipeline actually run?" gate. It does NOT make strong
claims about how many pairs survive — that depends on backend and threshold
tuning. The point is: the wiring works.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Re-use the mini-corpus generator from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_mini_corpus import generate as generate_mini_corpus  # noqa: E402

from pipelines.multimodal_runner import main as run_multimodal_runner  # noqa: E402


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _build_configs(tmp: Path) -> Path:
    """Write image / text / pipeline configs into ``tmp`` and return the pipeline path."""
    image_cfg_path = tmp / "image_config.yaml"
    text_cfg_path = tmp / "text_override.yaml"
    pipeline_cfg_path = tmp / "pipeline.yaml"

    _write_yaml(
        image_cfg_path,
        {
            "embedding": {
                "backend": "average_rgb",  # avoid CLIP download; keeps smoke fast
                "fallback": "average_rgb",
                "batch_size": 16,
                "device": "cpu",
            },
            "dedup": {
                "method": "qsemdedup",
                "eps": 0.10,
                "alpha": 0.7,
                "quality_metric": "file_size",
                "max_candidates": 1000,
            },
        },
    )

    _write_yaml(
        text_cfg_path,
        {
            "embedding": {
                "ngram_size": 3,
                "lowercase": True,
                "strip_non_alnum": True,
                "collapse_whitespace": True,
            },
            "dedup": {
                "method": "qsemdedup",
                "alpha": 0.7,
                "eps": 0.15,
                "quality_metric": "entropy",
                "n_clusters": 3,
                "min_cluster_size": 2,
                "sbert_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "sbert_device": "cpu",
                "sbert_batch_size": 16,
                "two_stage": False,
            },
        },
    )

    _write_yaml(
        pipeline_cfg_path,
        {
            "general": {
                "input_root": str(tmp / "mini" / "dataset"),
                "output_root": str(tmp / "outputs"),
                "temp_root": str(tmp / "artifacts"),
                "resume": False,
                "parallel_modalities": False,
                "parallel_workers": 1,
                "batch_size": 100,
                "joint_dedup": {"enabled": True, "pair_strategy": "stem"},
            },
            "logging": {"level": "INFO"},
            "executor": {"type": "local", "envs": {}},
            "data_quality": {"include_unusable_bins": False},
            "sorter": {
                "enabled": True,
                "manifest_name": "manifest.csv",
                "batch_size": 100,
                "move_files": False,
            },
            "image": {
                "enabled": True,
                "entrypoint": str(REPO_ROOT / "pipelines" / "modalities" / "image_runner.py"),
                "workdir": str(REPO_ROOT),
                "output_dir": str(tmp / "outputs" / "image"),
                "args": [],
                "env": {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                "config_file": str(image_cfg_path),
                "batch_size": 0,
                "max_workers": 2,
                "manifest_subset_count": 0,
            },
            "audio": {"enabled": False},
            "text": {
                "enabled": True,
                "entrypoint": str(REPO_ROOT / "pipelines" / "modalities" / "text_runner.py"),
                "workdir": str(REPO_ROOT),
                "output_dir": str(tmp / "outputs" / "text"),
                "args": [],
                "env": {"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                "config_file": str(text_cfg_path),
                "batch_size": 100,
                "max_workers": 1,
            },
            "report": {
                "summary_file": "summary.json",
                "markdown_file": "report.md",
            },
        },
    )

    return pipeline_cfg_path


def _latest_run_dir(artifacts_root: Path) -> Path:
    candidates = sorted(p for p in artifacts_root.iterdir() if p.is_dir() and p.name[:8].isdigit())
    if not candidates:
        raise RuntimeError(f"no run dir under {artifacts_root}")
    return candidates[-1]


def main() -> int:
    keep_temp = "--keep-temp" in sys.argv
    tmp_obj = tempfile.TemporaryDirectory(prefix="twork_e2e_") if not keep_temp else None
    tmp = Path(tmp_obj.name) if tmp_obj else Path(tempfile.mkdtemp(prefix="twork_e2e_keep_"))

    try:
        info = generate_mini_corpus(tmp / "mini")
        print(f"[smoke] mini corpus: {info}")

        config_path = _build_configs(tmp)
        print(f"[smoke] pipeline config at {config_path}")

        rc = run_multimodal_runner(["--config", str(config_path)])
        if rc != 0:
            print(f"[smoke] pipeline returned non-zero rc={rc}")
            return 1

        run_dir = _latest_run_dir(tmp / "artifacts")
        print(f"[smoke] run dir: {run_dir}")

        expected_stages = [
            "stage1_sorter",
            "stage2_image",
            "stage2_text",
            "stage3_joint_dedup",
        ]
        for stage in expected_stages:
            success = run_dir / stage / "_SUCCESS"
            if not success.exists():
                print(f"[smoke] FAIL: missing {success}")
                return 1
            print(f"  + {stage}/_SUCCESS")

        joint_summary_path = run_dir / "stage3_joint_dedup" / "joint_dedup_summary.json"
        if not joint_summary_path.exists():
            print(f"[smoke] FAIL: missing {joint_summary_path}")
            return 1
        joint = json.loads(joint_summary_path.read_text(encoding="utf-8"))
        stats = joint["stats"]
        print(f"[smoke] joint stats: {stats}")
        if not (1 <= stats["joint_keepers"] <= 10):
            print(f"[smoke] FAIL: joint_keepers={stats['joint_keepers']} outside [1,10]")
            return 1
        if stats.get("total_unique_pairs", 0) <= 0:
            print(f"[smoke] FAIL: total_unique_pairs={stats.get('total_unique_pairs')}")
            return 1
        if "image_keepers" not in stats or "text_keepers" not in stats:
            print(f"[smoke] FAIL: missing per-modality keeper counts in {stats}")
            return 1

        print("\n[smoke] e2e PASSED — full pipeline ran end-to-end.")
        if keep_temp:
            print(f"[smoke] temp dir kept at {tmp} (--keep-temp)")
        return 0
    finally:
        if tmp_obj is not None:
            tmp_obj.cleanup()
        elif not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
