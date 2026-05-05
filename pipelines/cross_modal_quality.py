"""Cross-modal consistency as a deduplication quality signal.

Plan A innovation #1 — Direction A:

Instead of using a unimodal quality heuristic (file size for images, Shannon
entropy for text, SNR for audio) inside the Q-SemDeDup score, we use the
**alignment between the item and its paired counterpart in another modality**.
Concretely:

- image item ``img_i`` paired with text ``txt_i`` (sidecar discovery)::

      Q_align(img_i) = cos( CLIP_image(img_i), CLIP_text(txt_i) )

- text item ``txt_i`` paired with image ``img_i``::

      Q_align(txt_i) = cos( CLIP_image(img_i), CLIP_text(txt_i) )

- audio item ``aud_i`` paired with text ``txt_i``::

      Q_align(aud_i) = cos( CLAP_audio(aud_i), CLAP_text(txt_i) )

Pairs whose modalities are tightly aligned are preferred during in-cluster
selection; loosely-aligned pairs are dropped first when a near-duplicate must
be removed. Compared to SemDeDup ("keep closest to centroid") and FairDeDup
(fairness-aware selection), using cross-modal alignment as the quality term
is novel and directly explains downstream MLLM gains: aligned image-text
pairs are exactly what LLaVA-style instruction tuning consumes.

Sidecar discovery follows the same convention as
:mod:`pipelines.joint_dedup`: ``<stem>.<ext>`` siblings in the same directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore


TEXT_EXTS = (".txt", ".md", ".json")
IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".heic", ".heif", ".avif",
)
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus")


# ---------------------------------------------------------------------------
# Sidecar discovery
# ---------------------------------------------------------------------------

def _find_sidecar(path: Path, candidate_exts: Iterable[str]) -> Optional[Path]:
    parent = path.parent
    stem = path.stem
    for ext in candidate_exts:
        cand = parent / f"{stem}{ext}"
        if cand.exists():
            return cand
        # Some datasets use uppercase extensions; try those too.
        cand_upper = parent / f"{stem}{ext.upper()}"
        if cand_upper.exists():
            return cand_upper
    return None


def find_text_sidecar(image_or_audio_path: Path) -> Optional[Path]:
    """Locate the text caption that pairs with this image / audio file."""
    return _find_sidecar(Path(image_or_audio_path), TEXT_EXTS)


def find_image_sidecar(text_path: Path) -> Optional[Path]:
    """Locate the image that pairs with this text file."""
    return _find_sidecar(Path(text_path), IMAGE_EXTS)


def find_audio_sidecar(text_path: Path) -> Optional[Path]:
    """Locate the audio clip that pairs with this text file."""
    return _find_sidecar(Path(text_path), AUDIO_EXTS)


def read_caption(text_path: Path, max_chars: int = 4096) -> str:
    """Read a sidecar caption file as plain text (truncated for safety)."""
    try:
        return Path(text_path).read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Alignment scorers (lazy heavy-dep imports)
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class _CLIPAlignmentModel:
    """Cached open_clip model for image-text alignment scoring."""

    _instance: Optional["_CLIPAlignmentModel"] = None

    def __init__(self, model_name: str, device: str) -> None:
        import open_clip  # type: ignore
        import torch  # type: ignore

        self._open_clip = open_clip
        self._torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @classmethod
    def get(cls, model_name: str, device: str) -> "_CLIPAlignmentModel":
        if cls._instance is None or cls._instance._cache_key != (model_name, device):
            inst = cls(model_name, device)
            inst._cache_key = (model_name, device)
            cls._instance = inst
        return cls._instance

    def encode_images(self, paths: Sequence[Path]) -> "np.ndarray":
        from PIL import Image  # type: ignore

        torch = self._torch
        feats = []
        with torch.no_grad():
            for p in paths:
                try:
                    img = Image.open(str(p)).convert("RGB")
                except Exception:
                    feats.append(None)
                    continue
                x = self.preprocess(img).unsqueeze(0).to(self.device)
                emb = self.model.encode_image(x)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
                feats.append(emb.cpu().numpy()[0])
        # Replace failed encodes with zero vectors so cosine returns 0.
        if not feats:
            return np.zeros((0, 1), dtype=np.float32)
        dim = next((f.shape[0] for f in feats if f is not None), 1)
        out = np.zeros((len(feats), dim), dtype=np.float32)
        for i, f in enumerate(feats):
            if f is not None:
                out[i] = f
        return out

    def encode_texts(self, captions: Sequence[str]) -> "np.ndarray":
        torch = self._torch
        if not captions:
            return np.zeros((0, 1), dtype=np.float32)
        tokens = self.tokenizer(list(captions)).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        return emb.cpu().numpy().astype(np.float32, copy=False)


def compute_clip_alignment(
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    model_name: str = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    device: str = "auto",
) -> "np.ndarray":
    """Per-pair CLIP cosine similarity between images and their captions.

    ``image_paths[i]`` is paired with ``captions[i]``. Returns a 1-D float32
    array of length ``len(image_paths)``. Missing or unloadable images get a
    score of 0.0.

    On the first call the open_clip model is loaded and cached at module
    level via :class:`_CLIPAlignmentModel`. Subsequent calls reuse it.
    """
    if np is None:
        raise RuntimeError("numpy is required for compute_clip_alignment")
    n = len(image_paths)
    if n != len(captions):
        raise ValueError("image_paths and captions must have the same length")
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    try:
        model = _CLIPAlignmentModel.get(model_name, _resolve_device(device))
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "open_clip + torch are required for cross_modal quality; install "
            "via `uv sync --extra image`."
        ) from exc

    img_emb = model.encode_images(list(image_paths))
    txt_emb = model.encode_texts(list(captions))
    if img_emb.shape[0] == 0 or txt_emb.shape[0] == 0:
        return np.zeros(n, dtype=np.float32)
    # Pair-wise cosine: rows of img_emb · matching rows of txt_emb.
    sims = (img_emb * txt_emb).sum(axis=1)
    return sims.astype(np.float32, copy=False)


class _CLAPAlignmentModel:
    """Cached msclap model for audio-text alignment scoring."""

    _instance: Optional["_CLAPAlignmentModel"] = None

    def __init__(self, device: str) -> None:
        from msclap import CLAP  # type: ignore

        # Re-use the same soundfile patch as the qsemdedup encoder.
        from audio.method.qsemdedup import _patch_msclap_read_audio  # type: ignore

        self.model = CLAP(use_cuda=(device == "cuda"))
        _patch_msclap_read_audio(self.model)
        self.device = device

    @classmethod
    def get(cls, device: str) -> "_CLAPAlignmentModel":
        if cls._instance is None or cls._instance.device != device:
            cls._instance = cls(device)
        return cls._instance

    def encode_audios(self, paths: Sequence[Path]) -> "np.ndarray":
        emb = self.model.get_audio_embeddings([str(p) for p in paths], resample=True)
        arr = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32, copy=False)

    def encode_texts(self, captions: Sequence[str]) -> "np.ndarray":
        emb = self.model.get_text_embeddings(list(captions))
        arr = np.asarray(emb, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32, copy=False)


def compute_clap_alignment(
    audio_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    device: str = "auto",
) -> "np.ndarray":
    """Per-pair CLAP cosine similarity between audio clips and their captions."""
    if np is None:
        raise RuntimeError("numpy is required for compute_clap_alignment")
    n = len(audio_paths)
    if n != len(captions):
        raise ValueError("audio_paths and captions must have the same length")
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    try:
        model = _CLAPAlignmentModel.get(_resolve_device(device))
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "msclap is required for audio cross_modal quality; install via "
            "`uv sync --extra audio`."
        ) from exc

    aud_emb = model.encode_audios(list(audio_paths))
    txt_emb = model.encode_texts(list(captions))
    if aud_emb.shape[0] == 0 or txt_emb.shape[0] == 0:
        return np.zeros(n, dtype=np.float32)
    return (aud_emb * txt_emb).sum(axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Pair-wise quality builders (one helper per modality)
# ---------------------------------------------------------------------------

def cross_modal_quality_for_images(
    image_paths: Sequence[Path],
    *,
    model_name: str = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    device: str = "auto",
    fallback: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """Look up text sidecars and return CLIP alignment scores per image.

    Images without a text sidecar receive ``fallback[i]`` if provided, else 0.
    """
    paired_idx: List[int] = []
    paired_imgs: List[Path] = []
    paired_caps: List[str] = []
    for i, p in enumerate(image_paths):
        sidecar = find_text_sidecar(Path(p))
        if sidecar is None:
            continue
        cap = read_caption(sidecar)
        if not cap.strip():
            continue
        paired_idx.append(i)
        paired_imgs.append(Path(p))
        paired_caps.append(cap)

    out = np.zeros(len(image_paths), dtype=np.float32) if fallback is None else fallback.astype(np.float32, copy=True)
    if not paired_idx:
        return out

    sims = compute_clip_alignment(paired_imgs, paired_caps, model_name=model_name, device=device)
    for slot, score in zip(paired_idx, sims):
        out[slot] = float(score)
    return out


def cross_modal_quality_for_texts(
    text_paths: Sequence[Path],
    *,
    model_name: str = "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    device: str = "auto",
    fallback: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """Look up image sidecars and return CLIP alignment scores per text."""
    paired_idx: List[int] = []
    paired_imgs: List[Path] = []
    paired_caps: List[str] = []
    for i, p in enumerate(text_paths):
        sidecar = find_image_sidecar(Path(p))
        if sidecar is None:
            continue
        cap = read_caption(Path(p))
        if not cap.strip():
            continue
        paired_idx.append(i)
        paired_imgs.append(sidecar)
        paired_caps.append(cap)

    out = np.zeros(len(text_paths), dtype=np.float32) if fallback is None else fallback.astype(np.float32, copy=True)
    if not paired_idx:
        return out

    sims = compute_clip_alignment(paired_imgs, paired_caps, model_name=model_name, device=device)
    for slot, score in zip(paired_idx, sims):
        out[slot] = float(score)
    return out


def cross_modal_quality_for_audio(
    audio_paths: Sequence[Path],
    *,
    device: str = "auto",
    fallback: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """Look up text sidecars and return CLAP alignment scores per audio clip."""
    paired_idx: List[int] = []
    paired_auds: List[Path] = []
    paired_caps: List[str] = []
    for i, p in enumerate(audio_paths):
        sidecar = find_text_sidecar(Path(p))
        if sidecar is None:
            continue
        cap = read_caption(sidecar)
        if not cap.strip():
            continue
        paired_idx.append(i)
        paired_auds.append(Path(p))
        paired_caps.append(cap)

    out = np.zeros(len(audio_paths), dtype=np.float32) if fallback is None else fallback.astype(np.float32, copy=True)
    if not paired_idx:
        return out

    sims = compute_clap_alignment(paired_auds, paired_caps, device=device)
    for slot, score in zip(paired_idx, sims):
        out[slot] = float(score)
    return out
