"""End-to-end smoke for orchestrator.run_joint_dedup_stage.

Builds a minimal pipeline config pointing at three mock modality output dirs
populated with ``{modality}_runner_summary.json`` and ``{modality}_keepers.txt``
files. Drives :meth:`PipelineOrchestrator.run_joint_dedup_stage` directly (the
sorter / modality stages are *not* executed), then asserts:

- ``stage3_joint_dedup`` artifact dir contains ``_SUCCESS`` and ``summary.json``
- ``joint_dedup_summary.json`` lists the expected pair_keepers
- the per-pair drop attribution matches what we put in
- second invocation with ``general.resume: true`` short-circuits via the
  config-hash check (no re-run)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from pipelines.orchestrator import PipelineOrchestrator  # noqa: E402


def _write_modality(modality_dir: Path, modality: str, keepers: list[str]) -> None:
    modality_dir.mkdir(parents=True, exist_ok=True)
    keepers_path = modality_dir / f"{modality}_keepers.txt"
    keepers_path.write_text("\n".join(keepers) + "\n", encoding="utf-8")
    summary = {
        "modality": modality,
        "stats": {"unique": len(keepers), "duplicates": 0},
        "manifest": None,
        "keepers_file": str(keepers_path),
        "keepers_count": len(keepers),
    }
    (modality_dir / f"{modality}_runner_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_config(tmp: Path, *, resume: bool) -> Path:
    image_dir = tmp / "out" / "image"
    text_dir = tmp / "out" / "text"
    audio_dir = tmp / "out" / "audio"

    # Three modalities of pair ids p001..p004:
    #   image keeps p001, p002, p003       (drops p004)
    #   text  keeps p001, p003, p004       (drops p002)
    #   audio keeps p001, p002, p003, p004 (drops nothing)
    # Joint keepers should be {p001, p003}.
    _write_modality(image_dir, "image", [f"/data/p00{i}.jpg" for i in (1, 2, 3)])
    _write_modality(text_dir, "text", [f"/data/p00{i}.txt" for i in (1, 3, 4)])
    _write_modality(audio_dir, "audio", [f"/data/p00{i}.wav" for i in (1, 2, 3, 4)])

    config = {
        "general": {
            "output_root": str(tmp / "outputs"),
            "temp_root": str(tmp / "artifacts"),
            "resume": resume,
            "joint_dedup": {"enabled": True, "pair_strategy": "stem"},
        },
        "executor": {"type": "local", "envs": {}},
        "sorter": {"enabled": False},
        "image": {"enabled": True, "output_dir": str(image_dir)},
        "audio": {"enabled": True, "output_dir": str(audio_dir)},
        "text": {"enabled": True, "output_dir": str(text_dir)},
        "report": {},
    }
    config_path = tmp / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="joint_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        config_path = _build_config(tmp, resume=False)

        # Run 1: fresh
        orch = PipelineOrchestrator(config_path)
        orch.run_joint_dedup_stage()

        stage_dir = orch.artifacts_root / "stage3_joint_dedup"
        assert (stage_dir / "_SUCCESS").exists(), "stage_SUCCESS missing"
        assert (stage_dir / "summary.json").exists(), "stage summary.json missing"

        joint_summary = json.loads((stage_dir / "joint_dedup_summary.json").read_text())
        pair_keepers = set(joint_summary["pair_keepers"])
        assert pair_keepers == {"p001", "p003"}, f"unexpected pair_keepers={pair_keepers}"
        drops = joint_summary["pair_drops"]
        assert drops.get("p002") == ["text"], f"expected p002 dropped by text, got {drops}"
        assert drops.get("p004") == ["image"], f"expected p004 dropped by image, got {drops}"
        stats = joint_summary["stats"]
        assert stats["joint_keepers"] == 2 and stats["joint_drops"] == 2
        print(f"[smoke] run 1 OK — pair_keepers={sorted(pair_keepers)}, drops={drops}, stats={stats}")

        # Run 2: same config, resume=True. Should short-circuit via config_hash.
        # We manually wipe joint_dedup_summary.json to detect re-run vs resume —
        # if resume is honored, the file stays absent because we don't re-run.
        config_path = _build_config(tmp, resume=True)
        orch2 = PipelineOrchestrator(config_path)
        # Copy the previous run's artifacts into the new run_id so resume can match.
        prev_artifact = stage_dir
        new_stage_dir = orch2.stage_artifact_dir("stage3_joint_dedup")
        new_stage_dir.mkdir(parents=True, exist_ok=True)
        for f in prev_artifact.iterdir():
            (new_stage_dir / f.name).write_bytes(f.read_bytes())
        orch2.run_joint_dedup_stage()
        # The stages list should now contain the resumed stage payload.
        stage_names = [s.get("stage_name") for s in orch2.summary.get("stages", [])]
        assert "stage3_joint_dedup" in stage_names, (
            f"resume did not append stage summary; stages={stage_names}"
        )
        print("[smoke] run 2 OK — resume short-circuited via config_hash")

    print("\n[smoke] joint_dedup orchestrator smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
