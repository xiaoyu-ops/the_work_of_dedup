"""Smoke for the audio Q-SemDeDup CLAP path.

Generates a tiny synthetic audio set (sine tones at 3 different frequencies +
near-duplicates with mild noise + an exact duplicate) and runs
:func:`audio.method.qsemdedup.deduplicate_qsemdedup` end-to-end with the real
CLAP encoder. Asserts that the pipeline returns sensible keeper counts and
that exact duplicates collapse.

This requires ``uv sync --extra audio`` (msclap + torchaudio + librosa). The
first invocation downloads CLAP weights (~500 MB), which is cached after.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audio.method.qsemdedup import (  # noqa: E402
    AudioQSemDedupConfig,
    deduplicate_qsemdedup,
)


def _tone(freq: float, sr: int = 48000, dur: float = 3.0, noise: float = 0.0,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    wav = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if noise > 0:
        wav = wav + rng.normal(0.0, noise, wav.shape).astype(np.float32)
    return wav


def _two_tone(freq_a: float, freq_b: float, sr: int = 48000, dur: float = 3.0) -> np.ndarray:
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    return (
        0.2 * np.sin(2 * np.pi * freq_a * t)
        + 0.2 * np.sin(2 * np.pi * freq_b * t)
    ).astype(np.float32)


CLIPS = [
    # name, waveform-fn-args
    ("a01.wav", lambda: _tone(440.0, seed=1)),
    ("a02.wav", lambda: _tone(440.0, noise=0.005, seed=2)),  # near-dup of a01
    ("a03.wav", lambda: _tone(440.0, seed=1)),                # exact-dup of a01
    ("b01.wav", lambda: _tone(880.0, seed=10)),
    ("b02.wav", lambda: _tone(880.0, noise=0.005, seed=11)),  # near-dup of b01
    ("c01.wav", lambda: _tone(1320.0, seed=20)),
    ("d01.wav", lambda: _two_tone(220.0, 660.0)),             # totally different
]


def main() -> int:
    sr = 48000
    with tempfile.TemporaryDirectory(prefix="audio_qsem_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        paths: list[Path] = []
        for name, fn in CLIPS:
            wav = fn()
            p = tmp / name
            sf.write(str(p), wav, sr)
            paths.append(p)
        print(f"[smoke] wrote {len(paths)} synthetic clips to {tmp}")

        cfg = AudioQSemDedupConfig(
            clap_backend="msclap",
            device="cpu",
            target_sr=sr,
            max_audio_seconds=3.0,
            n_clusters=3,
            min_cluster_size=2,
            eps=0.05,
            alpha=0.7,
            quality_metric="snr",
            two_stage=False,
        )

        print("[smoke] loading CLAP (first run downloads weights, may take a minute)")
        result = deduplicate_qsemdedup(paths, cfg)

        print("[smoke] result keys:", sorted(result.keys()))
        keepers = result["keepers"]
        duplicates = result["duplicates"]
        print(f"[smoke] keepers ({len(keepers)}):")
        for k in keepers:
            print(f"  + {k.name}")
        print(f"[smoke] duplicate groups ({len(duplicates)}):")
        for grp in duplicates:
            orig = Path(grp["original"]).name
            stage = grp.get("stage", "?")
            print(f"  - keeper={orig}  stage={stage}")
            for d in grp["duplicates"]:
                print(f"      ~ {Path(d['path']).name}  sim={d['similarity']:.3f}")

        keeper_names = {p.name for p in keepers}

        # The exact duplicate (a03 == a01) MUST collapse to a single keeper.
        assert "a01.wav" in keeper_names or "a03.wav" in keeper_names
        assert not ({"a01.wav", "a03.wav"}.issubset(keeper_names)), (
            "exact-dup a03 should collapse with a01 under CLAP"
        )

        # The dissimilar outlier (d01 — two-tone) should always survive.
        assert "d01.wav" in keeper_names, "two-tone outlier d01 should survive"

        print("\n[smoke] audio CLAP qsemdedup PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
