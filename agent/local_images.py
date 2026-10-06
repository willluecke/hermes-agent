"""Images a turn references by path on this host (``file://`` image parts).

Hermes Chat uploads a user's image straight to command-center, where the
sync store writes it to disk, and the model request then names that file
instead of carrying its base64: a 10 MB screenshot would be 13 MB of base64,
over the gateway's request cap and far over the app host's. Before this
(2026-10-01) the composer shrank every image to a 1280 px JPEG under 180 KB
to fit those caps, and site screenshots reached the model pixelated.

A ``file://`` URL is accepted only when its real path lies under one of
``agent.local_image_roots`` (default: the sync store's conversation-images
directory), is a regular file of an image type, and is at most
``MAX_LOCAL_IMAGE_BYTES``. Anything else is refused as an invalid image URL:
the API is authenticated, but a path outside those roots has no business in
a prompt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import unquote, urlparse

CONFIG_KEY = "local_image_roots"
DEFAULT_LOCAL_IMAGE_ROOTS = ["~/.hermes/conversation-images"]
MAX_LOCAL_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def local_image_roots(config: Optional[dict[str, Any]] = None) -> List[str]:
    """The directories a ``file://`` image may live under, resolved."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    raw = agent_cfg.get(CONFIG_KEY) if isinstance(agent_cfg, dict) else None
    if raw is None:
        raw = DEFAULT_LOCAL_IMAGE_ROOTS
    if isinstance(raw, str):
        raw = [raw]
    roots: List[str] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, str) or not entry.strip():
            continue
        try:
            roots.append(os.path.realpath(os.path.expanduser(entry.strip())))
        except OSError:
            continue
    return roots


def is_file_url(url: str) -> bool:
    return isinstance(url, str) and url.strip().lower().startswith("file://")


def _image_type(path: str, head: bytes) -> Optional[str]:
    """The media type the suffix claims, when the first bytes agree."""
    media_type = IMAGE_SUFFIXES.get(Path(path).suffix.lower())
    if media_type is None:
        return None
    if media_type == "image/png" and head.startswith(b"\x89PNG\r\n\x1a\n"):
        return media_type
    if media_type == "image/jpeg" and head.startswith(b"\xff\xd8\xff"):
        return media_type
    if media_type == "image/gif" and head[:6] in (b"GIF87a", b"GIF89a"):
        return media_type
    if media_type == "image/webp" and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return media_type
    return None


def local_image_path(url: str, config: Optional[dict[str, Any]] = None) -> str:
    """The on-disk path a ``file://`` image URL names, or ``ValueError`` saying why not."""
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("a local image must be a file:// URL on this host")
    raw_path = unquote(parsed.path or "")
    if not raw_path.startswith("/"):
        raise ValueError("a local image path must be absolute")
    roots = local_image_roots(config)
    if not roots:
        raise ValueError("no local image roots are configured (agent.local_image_roots)")
    try:
        real = os.path.realpath(raw_path)
    except OSError as exc:
        raise ValueError(f"local image path cannot be resolved: {exc}") from exc
    if not any(real == root or real.startswith(root.rstrip(os.sep) + os.sep) for root in roots):
        raise ValueError("local image is outside the configured image roots")
    if not os.path.isfile(real):
        raise ValueError("local image does not exist")
    size = os.path.getsize(real)
    if size == 0 or size > MAX_LOCAL_IMAGE_BYTES:
        raise ValueError(f"local image must be between 1 byte and {MAX_LOCAL_IMAGE_BYTES} bytes")
    with open(real, "rb") as handle:
        head = handle.read(16)
    if _image_type(real, head) is None:
        raise ValueError("local image must be a PNG, JPEG, WebP or GIF file whose bytes match its suffix")
    return real
