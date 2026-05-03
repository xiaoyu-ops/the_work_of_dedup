"""Unit-style smoke for sorter.sniff_magic.

Synthesizes minimal magic-byte headers for each format we expect to support
and asserts ``sniff_magic`` returns the right modality. Also drives
``determine_category`` end-to-end on temp files to confirm the relaxed
``STRICT_*`` whitelists let .webp / .mp3 / etc. survive past the post-check.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.sorter import (  # noqa: E402
    STRICT_AUDIO_EXTS,
    STRICT_IMAGE_EXTS,
    determine_category,
    sniff_magic,
)


SAMPLES = {
    # Images
    "png": (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image"),
    "jpeg_jfif": (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "image"),
    "jpeg_exif": (b"\xff\xd8\xff\xe1" + b"\x00" * 8, "image"),
    "gif87": (b"GIF87a" + b"\x00" * 8, "image"),
    "gif89": (b"GIF89a" + b"\x00" * 8, "image"),
    "webp": (b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 4, "image"),
    "tiff_le": (b"II*\x00" + b"\x00" * 8, "image"),
    "tiff_be": (b"MM\x00*" + b"\x00" * 8, "image"),
    "bmp": (b"BM" + b"\x00" * 12, "image"),
    "heic": (b"\x00\x00\x00\x18ftypheic" + b"\x00" * 4, "image"),
    "avif": (b"\x00\x00\x00\x18ftypavif" + b"\x00" * 4, "image"),
    "svg": (b"<?xml version=\"1.0\"?><svg xmlns=...", "image"),
    # Audio
    "wav": (b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 4, "audio"),
    "mp3_id3": (b"ID3\x04\x00" + b"\x00" * 10, "audio"),
    "mp3_frame_layer3": (b"\xff\xfb\x90\x44" + b"\x00" * 8, "audio"),  # MPEG1 layer III
    "aac_adts": (b"\xff\xf1" + b"\x00" * 12, "audio"),
    "flac": (b"fLaC" + b"\x00" * 12, "audio"),
    "ogg": (b"OggS" + b"\x00" * 12, "audio"),
    "m4a": (b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 4, "audio"),
    # Text-shaped JSON
    "json_obj": (b"{\"text\": \"hi\"}", "text"),
    "json_arr": (b"[\"a\", \"b\"]", "text"),
}


def main() -> int:
    fails: list[str] = []
    for name, (header, expected) in SAMPLES.items():
        got = sniff_magic(header)
        if got != expected:
            fails.append(f"  - {name}: expected={expected} got={got}")
    if fails:
        print("[smoke] sniff_magic failures:")
        for line in fails:
            print(line)
        return 1
    print(f"[smoke] sniff_magic: {len(SAMPLES)} formats all OK")

    # Strict whitelist check: assert .webp/.mp3 are in the relaxed sets.
    assert ".webp" in STRICT_IMAGE_EXTS, "STRICT_IMAGE_EXTS should contain .webp now"
    assert ".tiff" in STRICT_IMAGE_EXTS, "STRICT_IMAGE_EXTS should contain .tiff now"
    assert ".mp3" in STRICT_AUDIO_EXTS, "STRICT_AUDIO_EXTS should contain .mp3 now"
    assert ".flac" in STRICT_AUDIO_EXTS, "STRICT_AUDIO_EXTS should contain .flac now"
    print("[smoke] STRICT_* whitelists relaxed as expected")

    # End-to-end determine_category on temp files for a few representative formats.
    e2e = [
        (".webp", b"RIFF\x00\x00\x00\x00WEBP\x00\x00\x00\x00", "image"),
        (".mp3", b"\xff\xfb\x90\x44" + b"\x00" * 100, "audio"),
        (".flac", b"fLaC" + b"\x00" * 100, "audio"),
        (".jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 100, "image"),
        (".txt", b"hello world this is plain text\n", "text"),
        (".json", b"{\"caption\": \"a cat\"}", "text"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for ext, content, expected in e2e:
            p = Path(tmp) / f"sample{ext}"
            p.write_bytes(content)
            got = determine_category(str(p))
            assert got == expected, f"{ext}: expected={expected} got={got}"
            print(f"  + {ext} -> {got}")
    print("[smoke] determine_category e2e OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
