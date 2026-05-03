"""Audio instantiation of the unified Q-SemDeDup framework.

Embedding: CLAP (Contrastive Language-Audio Pretraining). Two backends are
           supported via lazy import — Microsoft's ``msclap`` and LAION's
           ``laion-clap`` (HF). The first one available is used unless
           ``config.clap_backend`` pins a specific choice.
Quality:   Signal-to-noise ratio (SNR) of the waveform, in dB. Falls back to
           effective duration (non-silent fraction of the clip) if SNR
           estimation is unavailable.
Selection: shared :mod:`pipelines.qsemdedup_core`.

Two execution modes mirror the text module:

- **single-stage**: encode every clip with CLAP, MiniBatchKMeans cluster, then
  run :func:`select_q_semdedup` per cluster.
- **two-stage**: build a coarse audio fingerprint (re-using the existing
  ``audio.method.spectrum_fingerprint`` peak-pair landmarks) and bucket via
  MinHash LSH; only encode buckets with >= 2 members with CLAP. This is the
  configuration recommended by plan A for large catalogues.

Heavy dependencies (``msclap`` / ``laion-clap`` / ``librosa`` / ``torch``) are
imported lazily so this module can be loaded without the audio extras
installed — useful at framework-construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
class AudioQSemDedupConfig:
    """Runtime knobs for audio Q-SemDeDup."""

    clap_backend: str = "auto"  # auto | msclap | laion_clap
    clap_model_path: Optional[str] = None  # backend-specific override
    device: str = "auto"  # auto | cpu | cuda
    target_sr: int = 48000  # CLAP models expect 48 kHz
    max_audio_seconds: float = 30.0  # cap per-clip cost; longer clips are truncated

    n_clusters: Optional[int] = None
    min_cluster_size: int = 2
    eps: float = 0.05
    alpha: float = 0.7
    quality_metric: str = "snr"  # snr | duration
    random_state: int = 42
    kmeans_max_iter: int = 100

    # Two-stage: spectrogram-fingerprint MinHash LSH coarse filter -> CLAP precision
    two_stage: bool = False
    lsh_threshold: float = 0.5
    lsh_num_perm: int = 128
    lsh_max_tokens: int = 200


# ---------------------------------------------------------------------------
# Audio I/O & quality
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_waveform(path: Path, target_sr: int, max_seconds: float):
    """Load a clip with librosa, mono, ``target_sr``, truncated to ``max_seconds``."""
    try:
        import librosa  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "librosa is required for audio qsemdedup; install via `uv sync --extra audio`"
        ) from exc

    duration = max(0.1, float(max_seconds))
    y, sr = librosa.load(str(path), sr=target_sr, mono=True, duration=duration)
    if y is None:
        return np.zeros(0, dtype=np.float32), target_sr
    return y.astype(np.float32, copy=False), int(sr)


def _estimate_snr_db(waveform: "np.ndarray") -> float:
    """Rough SNR estimate using a top-percentile / bottom-percentile energy ratio.

    No reference signal is available so this is a heuristic: the loudest 10% of
    frames is treated as "signal", the quietest 10% as "noise floor". The result
    correlates with perceptual quality enough for cross-clip ranking, which is
    all the Q-SemDeDup score needs.
    """
    if waveform.size == 0:
        return 0.0
    frame = 2048
    hop = 512
    if waveform.size < frame:
        return 0.0
    n_frames = 1 + (waveform.size - frame) // hop
    if n_frames <= 4:
        return 0.0
    energies = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        chunk = waveform[i * hop : i * hop + frame]
        energies[i] = float(np.mean(chunk * chunk))
    energies = np.sort(energies)
    noise = float(np.mean(energies[: max(1, n_frames // 10)]))
    signal = float(np.mean(energies[-max(1, n_frames // 10) :]))
    if noise <= 0 or signal <= 0:
        return 0.0
    return 10.0 * math.log10(signal / noise)


def _effective_duration(waveform: "np.ndarray", sr: int, threshold_db: float = -40.0) -> float:
    """Seconds of waveform whose frame energy is above ``threshold_db``."""
    if waveform.size == 0 or sr <= 0:
        return 0.0
    frame = 2048
    hop = 512
    if waveform.size < frame:
        return float(waveform.size) / float(sr)
    cutoff_lin = 10.0 ** (threshold_db / 10.0)
    peak_e = float(np.max(waveform * waveform)) or 1.0
    n_frames = 1 + (waveform.size - frame) // hop
    nonsilent = 0
    for i in range(n_frames):
        chunk = waveform[i * hop : i * hop + frame]
        if float(np.mean(chunk * chunk)) >= cutoff_lin * peak_e:
            nonsilent += 1
    return float(nonsilent * hop) / float(sr)


def _quality_score(waveform: "np.ndarray", sr: int, metric: str) -> float:
    if metric == "duration":
        return _effective_duration(waveform, sr)
    return _estimate_snr_db(waveform)


# ---------------------------------------------------------------------------
# CLAP encoding
# ---------------------------------------------------------------------------

def _patch_msclap_read_audio(clap_wrapper: Any) -> None:
    """Replace msclap's torchaudio-based ``read_audio`` with a soundfile loader.

    The default msclap path goes through ``torchaudio.load`` -> ``torchcodec``
    -> system FFmpeg. On a machine without ffmpeg installed (typical macOS
    out-of-the-box) this fails to load the shared libraries. ``soundfile``
    handles WAV / FLAC / OGG natively via libsndfile, which is bundled in the
    soundfile wheel — no system ffmpeg needed. For MP3/AAC/M4A users still
    need ffmpeg, but plan A's audio runs use WAV.
    """
    try:
        import soundfile as sf  # type: ignore
        import torch  # type: ignore
        import torchaudio.transforms as T  # type: ignore
    except ImportError:
        return

    def _read_audio_soundfile(self, audio_path, resample: bool = True):
        wav, sample_rate = sf.read(str(audio_path), always_2d=False)
        if wav.ndim == 2:
            # Mix down to mono.
            wav = wav.mean(axis=1)
        tensor = torch.from_numpy(wav).float().unsqueeze(0)
        target_sr = self.args.sampling_rate
        if resample and target_sr != sample_rate:
            tensor = T.Resample(sample_rate, target_sr)(tensor)
            sample_rate = target_sr
        return tensor, sample_rate

    import types

    clap_wrapper.read_audio = types.MethodType(_read_audio_soundfile, clap_wrapper)


class _CLAPEncoder:
    """Thin adapter that hides backend (msclap / laion-clap) selection."""

    def __init__(self, config: AudioQSemDedupConfig):
        self.config = config
        self.backend = None
        self._impl = None

    def _try_msclap(self):
        try:
            from msclap import CLAP  # type: ignore
        except ImportError:
            return None
        device = _resolve_device(self.config.device)
        use_cuda = device == "cuda"
        model_kwargs: Dict[str, Any] = {"use_cuda": use_cuda}
        if self.config.clap_model_path:
            model_kwargs["model_fp"] = self.config.clap_model_path
        instance = CLAP(**model_kwargs)
        _patch_msclap_read_audio(instance)
        return ("msclap", instance)

    def _try_laion(self):
        try:
            import laion_clap  # type: ignore
        except ImportError:
            return None
        device = _resolve_device(self.config.device)
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        if self.config.clap_model_path:
            model.load_ckpt(self.config.clap_model_path)
        else:
            model.load_ckpt()
        return ("laion_clap", model)

    def load(self) -> None:
        order = []
        choice = (self.config.clap_backend or "auto").lower()
        if choice == "auto":
            order = [self._try_msclap, self._try_laion]
        elif choice == "msclap":
            order = [self._try_msclap]
        elif choice in {"laion_clap", "laion"}:
            order = [self._try_laion]
        else:
            raise ValueError(f"Unknown CLAP backend: {self.config.clap_backend}")

        last_exc: Optional[Exception] = None
        for fn in order:
            try:
                got = fn()
            except Exception as exc:  # pragma: no cover - depends on env
                last_exc = exc
                continue
            if got is not None:
                self.backend, self._impl = got
                return
        raise RuntimeError(
            "No CLAP backend available. Install one of `msclap` or `laion-clap` via "
            "`uv sync --extra audio` to enable audio qsemdedup."
        ) from last_exc

    def encode(self, paths: Sequence[Path]) -> "np.ndarray":
        if self._impl is None:
            self.load()
        if self.backend == "msclap":
            emb = self._impl.get_audio_embeddings([str(p) for p in paths], resample=True)
            arr = np.asarray(emb, dtype=np.float32)
        elif self.backend == "laion_clap":
            arr = self._impl.get_audio_embedding_from_filelist(
                x=[str(p) for p in paths], use_tensor=False
            )
            arr = np.asarray(arr, dtype=np.float32)
        else:
            raise RuntimeError("CLAP encoder not initialized")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Two-stage signature: coarse band-energy fingerprint
# ---------------------------------------------------------------------------

def _band_energy_tokens(
    waveform: "np.ndarray",
    sr: int,
    cap: int,
    n_bands: int = 8,
) -> List[bytes]:
    """Compute coarse 8-band log-energy quantization tokens from a waveform.

    Two clips that share spectro-temporal structure share a large fraction of
    these tokens, which is enough for the LSH coarse-filter role: the
    precision stage (CLAP) re-scores everything that survives. A Shazam-style
    peak-pair fingerprint is a future option, but for plan A's two-stage
    pipeline this representation hits the right cost/recall trade-off
    (no model, no GPU) on the order of microseconds per clip.
    """
    if waveform is None or sr <= 0 or waveform.size == 0:
        return []
    win = sr  # 1-second window
    if waveform.size < win:
        return []
    out: List[bytes] = []
    for start in range(0, waveform.size - win + 1, win):
        if len(out) >= cap:
            break
        chunk = waveform[start : start + win]
        spec = np.abs(np.fft.rfft(chunk))
        band_size = max(1, len(spec) // n_bands)
        energies = []
        for b in range(n_bands):
            seg = spec[b * band_size : (b + 1) * band_size]
            energies.append(int(round(float(np.log1p(seg.sum())))))
        out.append(":".join(str(e) for e in energies).encode("utf-8"))
    return out


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

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
    config: AudioQSemDedupConfig,
) -> Dict[str, Any]:
    """Run audio Q-SemDeDup over a sequence of audio file paths.

    Output mirrors the schema produced by other audio dedup methods:
    ``keepers``, ``duplicates``, ``duplicate_count``, ``skipped``.
    """
    if np is None:
        raise RuntimeError("numpy is required for audio qsemdedup")
    n = len(paths)
    if n == 0:
        return {"keepers": [], "duplicates": [], "duplicate_count": 0, "skipped": 0}

    # 1. Single per-file pass: load each waveform once with librosa, compute
    #    quality score, and (when two_stage) precompute the LSH band-energy
    #    tokens. Avoids re-loading the file for the signature step. CLAP loads
    #    its own copy internally — that read we cannot avoid without rewriting
    #    the encoder, so the lower bound is 2x I/O per file (us + CLAP).
    qualities = np.zeros(n, dtype=np.float32)
    pre_signatures: Optional[List[List[bytes]]] = (
        [[] for _ in range(n)] if config.two_stage else None
    )
    sig_cap = max(1, config.lsh_max_tokens)
    for i, p in enumerate(paths):
        try:
            wav, sr = _load_waveform(Path(p), config.target_sr, config.max_audio_seconds)
            qualities[i] = float(_quality_score(wav, sr, config.quality_metric))
            if pre_signatures is not None:
                pre_signatures[i] = _band_energy_tokens(wav, sr, sig_cap)
        except Exception:
            qualities[i] = 0.0
    qualities_norm = normalize_quality(qualities)

    threshold = 1.0 - float(config.eps)
    keep_flags = np.zeros(n, dtype=bool)
    duplicates_out: List[Dict[str, object]] = []
    duplicate_count = 0

    encoder = _CLAPEncoder(config)

    if config.two_stage:
        # Hand pre-computed tokens to lsh_buckets via a tiny closure that just
        # reads from the index → tokens table. Passing indices (0..n-1) as the
        # ``items`` keeps the bookkeeping in one numeric space.
        sig_lookup = pre_signatures or []

        def _signature_from_index(idx: int):
            return iter(sig_lookup[idx])

        buckets = lsh_buckets(
            list(range(n)),
            _signature_from_index,
            threshold=config.lsh_threshold,
            num_perm=config.lsh_num_perm,
        )
        nontrivial = [b for b in buckets if len(b) >= max(2, config.min_cluster_size)]
        nontrivial_set = {id(b) for b in nontrivial}
        for bucket in buckets:
            if id(bucket) in nontrivial_set:
                continue
            for idx in bucket:
                keep_flags[idx] = True

        if nontrivial:
            embed_global_idx = [i for b in nontrivial for i in b]
            sub_paths = [paths[i] for i in embed_global_idx]
            sub_embeddings = encoder.encode(sub_paths)
            global_to_sub = {gi: si for si, gi in enumerate(embed_global_idx)}

            for bucket_id, bucket in enumerate(nontrivial):
                rows = [global_to_sub[i] for i in bucket]
                member_feats = sub_embeddings[rows]
                member_qual = qualities_norm[bucket]
                result = select_q_semdedup(
                    bucket, member_feats, member_qual, threshold, config.alpha
                )
                for k in result["keep_indices"]:
                    keep_flags[k] = True
                for keeper_global, dup_list in result["dup_groups"].items():
                    duplicates_out.append(
                        _emit_dup_record(
                            paths, keeper_global, dup_list, threshold, bucket_id,
                            f"lsh+clap[{encoder.backend or 'auto'}]"
                        )
                    )
                    duplicate_count += len(dup_list)
    else:
        embeddings = encoder.encode(list(paths))
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
                        paths, keeper_global, dup_list, threshold, cluster_id,
                        f"kmeans+clap[{encoder.backend or 'auto'}]"
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
