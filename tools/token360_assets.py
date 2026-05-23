"""Token360 asset upload helpers.

Why this exists
---------------
Token360's gateway sits behind an AWS WAF that rejects ``/images/generations``
and ``/chat/completions`` request bodies containing inline base64 image data
larger than a few KB (HTTP 403, HTML body, no JSON). The supported pattern is:

    1. POST /v1/asset-groups            -> assetGroupId  (re-use across calls)
    2. POST /v1/assets    (multipart)   -> assetId       (ua_xxx)
    3. GET  /v1/assets/{id}             -> wait until status == "active"
    4. Use ``asset://{assetId}`` in the request body wherever an image URL is
       expected (e.g. ``frame_images[].image_url.url`` for video,
       ``reference_images[]`` for image-to-image with Nano Banana Pro).

This module wraps that flow with a SHA256-keyed on-disk cache, so the same
local file is uploaded once per group across runs.

Usage
-----
    from tools.token360_assets import Token360AssetUploader

    uploader = Token360AssetUploader()           # reads .env
    uri = await uploader.upload("/tmp/portrait.png")
    # uri == "asset://ua_3f9c..."
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import asyncio
from pathlib import Path
from typing import Optional

import aiohttp


# Default mime resolution for common extensions
_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
}


def _guess_mime(path: str) -> str:
    return _MIME.get(Path(path).suffix.lower(), "application/octet-stream")


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Token360AssetUploader:
    """Upload local files to Token360 and return ``asset://...`` URIs.

    The uploader maintains a single GENERIC asset group per (uploader name)
    and caches (sha256 -> assetId) mappings on disk to avoid re-upload.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        group_name: str = "vimax-refs",
        cache_dir: str = ".working_dir/_token360_assets_cache",
        group_kind: str = "GENERIC",
    ):
        self.api_key = api_key or os.environ.get("TOKEN360_API_KEY", "")
        if not self.api_key:
            raise ValueError("TOKEN360_API_KEY missing; set it in .env.")
        self.base_url = (
            base_url or os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1")
        ).rstrip("/")
        self.group_name = group_name
        self.group_kind = group_kind

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._group_cache = self.cache_dir / "group.json"
        self._asset_cache = self.cache_dir / "assets.json"
        self._assets: dict[str, str] = (
            json.loads(self._asset_cache.read_text()) if self._asset_cache.exists() else {}
        )
        self._group_id: Optional[str] = None
        self._group_lock = asyncio.Lock()

    # ---------------------------------------------------------------- group

    async def _ensure_group(self) -> str:
        if self._group_id:
            return self._group_id
        async with self._group_lock:
            if self._group_id:
                return self._group_id
            if self._group_cache.exists():
                try:
                    gid = json.loads(self._group_cache.read_text())["assetGroupId"]
                    # Trust the cached group; if it's been deleted server-side a
                    # later upload will 404 and we'll fall through to create.
                    self._group_id = gid
                    return gid
                except Exception:
                    pass

            url = f"{self.base_url}/asset-groups"
            payload = {"name": self.group_name, "groupKind": self.group_kind}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    body = await r.json()
            if r.status >= 400 or "data" not in body:
                raise RuntimeError(f"Token360 asset-group create failed ({r.status}): {body}")
            gid = body["data"]["assetGroupId"]
            self._group_id = gid
            self._group_cache.write_text(json.dumps({"assetGroupId": gid}))
            logging.info(f"[Token360] created asset group {gid} ({self.group_name})")
            return gid

    # ---------------------------------------------------------------- upload

    async def upload(self, path: str, asset_name: Optional[str] = None) -> str:
        """Upload ``path`` if not cached, return ``asset://...`` URI."""
        key = _sha256_of_file(path)
        if key in self._assets:
            return f"asset://{self._assets[key]}"

        gid = await self._ensure_group()
        url = f"{self.base_url}/assets"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with open(path, "rb") as f:
            blob = f.read()

        form = aiohttp.FormData()
        form.add_field("groupId", gid)
        form.add_field("name", asset_name or Path(path).name)
        form.add_field("file", blob, filename=Path(path).name, content_type=_guess_mime(path))

        async with aiohttp.ClientSession() as s:
            async with s.post(url, data=form, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as r:
                body = await r.json()
        if r.status >= 400 or "data" not in body:
            raise RuntimeError(f"Token360 asset upload failed ({r.status}): {body}")

        asset_id = body["data"]["assetId"]
        await self._wait_active(asset_id)

        self._assets[key] = asset_id
        self._asset_cache.write_text(json.dumps(self._assets, indent=2))
        logging.info(f"[Token360] uploaded {path} -> asset://{asset_id}")
        return f"asset://{asset_id}"

    async def _wait_active(self, asset_id: str, timeout: float = 300.0, poll: float = 2.0) -> None:
        url = f"{self.base_url}/assets/{asset_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        elapsed = 0.0
        while elapsed < timeout:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    body = await r.json()
            status = str((body.get("data") or body).get("status", "")).lower()
            if status in ("active", "ready"):
                return
            if status in ("failed", "error"):
                raise RuntimeError(f"Asset {asset_id} failed: {body}")
            await asyncio.sleep(poll)
            elapsed += poll
        raise TimeoutError(f"Asset {asset_id} not active after {timeout}s")
