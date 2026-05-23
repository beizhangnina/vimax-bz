"""Video generator backed by Token360's POST /videos endpoint.

Default model is ``seedance-2.0`` (BytePlus Dreamina Seedance 2.0, multimodal:
text / first frame / first+last frame / reference images). Token360 also lists
``seedance-2.0-fast`` for lower-latency runs.

Frame image handling (observed against the live gateway):

* **Local file path** -> JPEG-downsampled to <=1280 px and inlined as a
  base64 data URL inside ``frame_images[].image_url.url``. The /videos
  endpoint accepts inline base64 for plain content, unlike /images/generations
  which has a WAF that blocks it.
* **``asset://...`` URI** -> passed through verbatim. Use this for
  RealFace / Virtual Portrait assets (the only asset kinds Seedance accepts
  for frame_images; generic USER_UPLOAD assets return
  "Invalid or unauthorized asset references").
* **``http(s)://`` URL** -> passed through verbatim.

Reference image semantics:

* 0 references -> text-to-video
* 1 reference  -> first-frame I2V (``frame_type: first_frame``)
* 2 references -> first + last frame interpolation
* 3+ references -> first_frame + ``input_references`` array (style/identity refs)

For verified-human consistency, create a RealFace group via Token360's H5
verification flow (see docs/models/real-face), then pass the resulting URIs
explicitly via the ``portrait_asset_uris=[...]`` keyword arg — they go into
``input_references``.
"""

from __future__ import annotations

import io
import os
import base64
import logging
import asyncio
from typing import List, Literal, Optional

import aiohttp
from PIL import Image

from interfaces.video_output import VideoOutput

ALLOWED_DURATIONS = {3, 4, 5, 6, 7, 8, 9, 10, 12}


def _frame_to_b64_url(path: str, max_dim: int = 1280, jpeg_quality: int = 88) -> str:
    """JPEG-downsample a local image and return a base64 data URL.

    1280-px JPEG is the largest size Seedance 2.0 input frames usefully
    consume; further upscaling would only inflate the request without
    affecting output resolution (controlled by ``size``).
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _to_frame_url(ref: str) -> str:
    """Convert any input form (local path / asset:// URI / http(s) URL) to
    the string usable in ``frame_images[].image_url.url``."""
    if ref.startswith(("asset://", "http://", "https://")):
        return ref
    return _frame_to_b64_url(ref)


class VideoGeneratorSeedanceToken360API:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        t2v_model: str = "seedance-2.0",
        ff2v_model: str = "seedance-2.0",
        flf2v_model: str = "seedance-2.0",
        rate_limiter=None,
    ):
        self.api_key = api_key or os.environ.get("TOKEN360_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Token360 API key missing. Pass api_key=... or set TOKEN360_API_KEY in .env."
            )
        self.base_url = (
            base_url or os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1")
        ).rstrip("/")
        self.submit_url = f"{self.base_url}/videos"
        self.t2v_model = t2v_model
        self.ff2v_model = ff2v_model
        self.flf2v_model = flf2v_model
        self.rate_limiter = rate_limiter

    def _pick_model(self, n_refs: int) -> str:
        if n_refs == 0:
            return self.t2v_model
        if n_refs == 1:
            return self.ff2v_model
        if n_refs == 2:
            return self.flf2v_model
        return self.ff2v_model  # 3+ refs still use first-frame model

    def _build_payload(
        self,
        prompt: str,
        reference_image_paths: List[str],
        portrait_asset_uris: List[str],
        duration: int,
        resolution: str,
        aspect_ratio: str,
    ) -> dict:
        model = self._pick_model(len(reference_image_paths))
        ref_urls = [_to_frame_url(p) for p in reference_image_paths]

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }

        frame_images = []
        if len(ref_urls) >= 1:
            frame_images.append({
                "type": "image_url",
                "frame_type": "first_frame",
                "image_url": {"url": ref_urls[0]},
            })
        if len(ref_urls) >= 2:
            frame_images.append({
                "type": "image_url",
                "frame_type": "last_frame",
                "image_url": {"url": ref_urls[1]},
            })
        if frame_images:
            payload["frame_images"] = frame_images

        # 3+ references: extras become style/identity references
        input_refs = []
        if len(ref_urls) > 2:
            input_refs.extend(
                {"type": "image_url", "image_url": {"url": u}} for u in ref_urls[2:]
            )
        # RealFace / Virtual Portrait asset URIs (caller supplies; only asset:// URIs accepted here)
        if portrait_asset_uris:
            input_refs.extend(
                {"type": "image_url", "image_url": {"url": u}} for u in portrait_asset_uris
            )
        if input_refs:
            payload["input_references"] = input_refs

        return payload

    async def _create_task(self, payload: dict) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        logging.info(
            f"[Token360] Submitting video task: model={payload['model']} "
            f"duration={payload.get('duration')} "
            f"frames={len(payload.get('frame_images', []))} "
            f"refs={len(payload.get('input_references', []))}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.submit_url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                body = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"Token360 /videos POST failed ({response.status}): {body}"
                    )
        task_id = (
            body.get("id")
            or body.get("task_id")
            or (body.get("data") or {}).get("id")
        )
        if not task_id:
            raise RuntimeError(f"No task id in /videos response: {body}")
        logging.info(f"[Token360] Video task created: {task_id}")
        return task_id

    async def _poll_task(self, task_id: str, timeout: float = 1200.0, poll: float = 5.0) -> str:
        query_url = f"{self.base_url}/videos/{task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        elapsed = 0.0
        while elapsed < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        query_url, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        body = await response.json()
            except aiohttp.ClientError as e:
                logging.warning(f"[Token360] poll {task_id} network error: {e}; retrying in 2s")
                await asyncio.sleep(2)
                elapsed += 2
                continue

            status = (body.get("status") or "").lower()
            if status in ("completed", "succeeded"):
                return _extract_video_url(body)
            if status in ("failed", "error"):
                raise RuntimeError(f"Token360 video task {task_id} failed: {body}")

            logging.info(f"[Token360] task {task_id} status={status or 'unknown'}; next poll in {poll}s")
            await asyncio.sleep(poll)
            elapsed += poll

        raise TimeoutError(f"Token360 video task {task_id} did not finish within {timeout}s")

    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
        resolution: Literal["480p", "720p", "1080p"] = "720p",
        aspect_ratio: str = "16:9",
        fps: Literal[16, 24] = 24,
        duration: int = 5,
        portrait_asset_uris: Optional[List[str]] = None,
        **kwargs,
    ) -> VideoOutput:
        if duration not in ALLOWED_DURATIONS:
            raise ValueError(f"duration must be one of {sorted(ALLOWED_DURATIONS)}; got {duration}")

        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        payload = self._build_payload(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
            portrait_asset_uris=portrait_asset_uris or [],
            duration=duration,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
        )

        task_id = await self._create_task(payload)
        video_url = await self._poll_task(task_id)
        logging.info(f"[Token360] Video URL: {video_url}")
        return VideoOutput(fmt="url", ext="mp4", data=video_url)


def _extract_video_url(body: dict) -> str:
    for key in ("video_url", "url"):
        v = body.get(key)
        if isinstance(v, str) and v:
            return v
    content = body.get("content") or body.get("video") or body.get("data") or {}
    if isinstance(content, dict):
        for key in ("video_url", "url"):
            v = content.get(key)
            if isinstance(v, str) and v:
                return v
    raise RuntimeError(f"Could not extract video URL from Token360 response: {body}")
