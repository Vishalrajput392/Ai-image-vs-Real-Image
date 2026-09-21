"""
backend/metadata_check.py
--------------------------
Lightweight, SEPARATE metadata hint — NOT part of the model's
prediction and never blended into ai_generated_probability. Looks for
known AI-tool signatures in JPEG EXIF tags or PNG text chunks (many
tools like Automatic1111/ComfyUI embed full generation parameters
directly in PNG metadata).

Explicitly a side-hint only:
    - Easily stripped by screenshots, re-saves, social-media re-uploads
    - Easily spoofed/forged
    - Absence of a match means NOTHING — most AI tools don't tag at all
    - Presence of a match is suggestive, not proof

Dependencies:
    exifread (pure Python, no Pillow), zlib + struct (stdlib)
"""

import io
import struct
import zlib

import exifread

AI_KEYWORDS = [
    "meta ai", "midjourney", "dall-e", "dall·e", "stable diffusion",
    "stablediffusion", "adobe firefly", "firefly", "leonardo.ai",
    "leonardo ai", "flux", "comfyui", "automatic1111", "c2pa",
    "ai-generated", "ai generated", "generative ai", "runway",
    "imagen", "gemini", "sdxl", "negative prompt", "cfg scale",
]


def _scan_text_for_keywords(text: str) -> list:
    text_lower = text.lower()
    return sorted({kw for kw in AI_KEYWORDS if kw in text_lower})


def _check_jpeg_exif(raw_bytes: bytes) -> dict:
    try:
        tags = exifread.process_file(io.BytesIO(raw_bytes), details=False)
    except Exception:
        return {"matched_keywords": [], "matched_fields": {}}

    matched_fields = {}
    for tag_name, value in (tags or {}).items():
        val_str = str(value)
        if _scan_text_for_keywords(val_str):
            matched_fields[str(tag_name)] = val_str[:200]

    all_hits = sorted({kw for v in matched_fields.values() for kw in _scan_text_for_keywords(v)})
    return {"matched_keywords": all_hits, "matched_fields": matched_fields}


def _check_png_text_chunks(raw_bytes: bytes) -> dict:
    matched_fields = {}
    if raw_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        return {"matched_keywords": [], "matched_fields": {}}

    pos = 8
    length = len(raw_bytes)

    while pos + 8 <= length:
        chunk_len = struct.unpack(">I", raw_bytes[pos:pos + 4])[0]
        chunk_type = raw_bytes[pos + 4:pos + 8].decode("ascii", errors="ignore")
        data_start = pos + 8
        data_end = data_start + chunk_len
        if data_end > length:
            break
        data = raw_bytes[data_start:data_end]

        try:
            if chunk_type == "tEXt":
                keyword, _, text = data.partition(b"\x00")
                text_str = text.decode("latin-1", errors="ignore")
            elif chunk_type == "zTXt":
                keyword, rest = data.split(b"\x00", 1)
                text_str = zlib.decompress(rest[1:]).decode("latin-1", errors="ignore")
            elif chunk_type == "iTXt":
                parts = data.split(b"\x00", 4)
                keyword = parts[0]
                compressed_flag = parts[1] if len(parts) > 1 else b"\x00"
                text_bytes = parts[4] if len(parts) > 4 else b""
                text_str = (
                    zlib.decompress(text_bytes).decode("utf-8", errors="ignore")
                    if compressed_flag == b"\x01"
                    else text_bytes.decode("utf-8", errors="ignore")
                )
            else:
                pos = data_end + 4
                continue
        except Exception:
            pos = data_end + 4
            continue

        hits = _scan_text_for_keywords(text_str)
        if hits:
            matched_fields[keyword.decode("ascii", errors="ignore")] = text_str[:200]

        pos = data_end + 4  # skip the trailing 4-byte CRC too

    all_hits = sorted({kw for v in matched_fields.values() for kw in _scan_text_for_keywords(v)})
    return {"matched_keywords": all_hits, "matched_fields": matched_fields}


def check_metadata_hint(raw_bytes: bytes) -> dict:
    """
    Returns a side-hint dict. Never used to change ai_generated_probability
    or predicted_label — purely informational, shown separately.
    """
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        result = _check_png_text_chunks(raw_bytes)
    else:
        result = _check_jpeg_exif(raw_bytes)

    return {
        "checked": True,
        "matched_keywords": result["matched_keywords"],
        "matched_fields": result["matched_fields"],
        "note": (
            "Informational only — not used by the model. Absence of a match "
            "does not mean the image is real; presence does not guarantee it "
            "is AI-generated (metadata is easily stripped or spoofed)."
        ),
    }
