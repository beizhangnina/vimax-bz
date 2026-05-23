"""Video generator backed by Token360's POST /videos endpoint.

Plugs into ViMax's VideoGenerator protocol via the async
``generate_single_video(prompt, reference_image_paths, **kwargs) -> VideoOutput``
method.

Token360 wraps multiple video providers behind one OpenAI-style gateway. The
default model is ``seedance-2.0-fast`` (BytePlus Seedance), which accepts
``first_frame``, ``last_frame``, and reference-image inputs.

Reference image semantics (matching ViMax's existing Yunwu Seedance adapter):

* 0 references -> text-to-video
* 1 reference  -> first-frame I2V
* 2 references -> first-frame + last-frame interpolation

Auth/base_url fall back to ``TOKEN360_API_KEY`` / ``TOKEN360_BASE_URL`` env
vars when constructor args are empty.

API shape (confirmed against the working aivideo client at
``/Users/beizhang/Documents/AI_Dev/ai_video/src/aivideo/generate/video.py``)::

  POST {base_url}/videos
  Authorization: Bearer <api_key>
  {
    "model": "seedance-2.0-fast",
    "prompt": "...",
    "seconds": 5,
    "size": "1024x1024",
    "frame_images": [
      {"type": "image_url", "frame_type": "first_frame",
       "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }
  -> 200 {"id": "task_..."}

  GET {base_url}/videos/{task_id}
  -> {"status": "completed" | "failed" | other, "video_url"|"url"|"data.url": "..."}
"""

from __future__ import annotations

import os
import io
import base64
import logging
import asyncio
from typing import List, Literal, Optional

import aiohttp
from PIL import Image

from interfaces.video_output import VideoOutput


def _frame_to_b64_url(path: str, max_dim: int = 1280, jpeg_quality: int = 88) -> str:
    """Encode a first/last-frame image as base64 data URL, downsampled to
    keep video-task POST bodies under Token360's ~1 MB AWS ELB limit.

    1280-px JPEG is the largest size Seedance 2.0 input frames usefully
    consume; further upscaling would only inflate the request without
    affecting the rendered output resolution (controlled by `size`).
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

ALLOWED_DURATIONS = {3, 4, 5, 6, 7, 8, 9, 10, 12}


class VideoGeneratorSeedanceToken360API:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        t2v_model: str = "seedance-2.0-fast",
        ff2v_model: str = "seedance-2.0-fast",
        flf2v_model: str = "seedance-2.0-fast",
        rate_limiter=None,
    ):
        self.api_key = api_key or os.environ.get("TOKEN360_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Token360 API key missing. Pass api_key=... or set TOKEN360_API_KEY in .env."
            )
        self.base_url = (base_url or os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1")).rstrip("/")
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
        raise ValueError("reference_image_paths must contain 0, 1, or 2 images.")

    async def _create_task(
        self,
        prompt: str,
        reference_image_paths: List[str],
        duration: int,
        size: str,
    ) -> str:
        model = self._pick_model(len(reference_image_paths))
        logging.info(f"[Token360] Submitting video task: model={model} refs={len(reference_image_paths)} dur={duration}s")

        payload: dict = {
            "model": model,
            "prompt": prompt,
            "seconds": duration,
            "size": size,
        }

        frame_images = []
        if len(reference_image_paths) >= 1:
            frame_images.append({
                "type": "image_url",
                "frame_type": "first_frame",
                "image_url": {"url": _frame_to_b64_url(reference_image_paths[0])},
            })
        if len(reference_image_paths) >= 2:
            frame_images.append({
                "type": "image_url",
                "frame_type": "last_frame",
                "image_url": {"url": _frame_to_b64_url(reference_image_paths[1])},
            })
        if frame_images:
            payload["frame_images"] = frame_images

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.submit_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as response:
                        response_json = await response.json()
                        if response.status >= 400:
                            raise RuntimeError(f"Token360 /videos POST failed ({response.status}): {response_json}")
                        task_id = (
                            response_json.get("id")
                            or response_json.get("task_id")
                            or response_json.get("data", {}).get("id")
                        )
                        if not task_id:
                            raise RuntimeError(f"No task id in /videos response: {response_json}")
            except aiohttp.ClientError as e:
                logging.error(f"[Token360] Network error submitting video task: {e}. Retrying in 2s...")
                await asyncio.sleep(2)
                continue
            break

        logging.info(f"[Token360] Video task created: {task_id}")
        return task_id

    async def _poll_task(self, task_id: str, timeout: float = 1200.0, poll: float = 5.0) -> str:
        """Poll until the task completes; return the video URL."""
        query_url = f"{self.base_url}/videos/{task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        elapsed = 0.0
        while elapsed < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(query_url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        body = await response.json()
            except aiohttp.ClientError as e:
                logging.error(f"[Token360] Network error polling task {task_id}: {e}. Retrying in 2s...")
                await asyncio.sleep(2)
                elapsed += 2
                continue

            status = (body.get("status") or "").lower()
            if status in ("completed", "succeeded"):
                return _extract_video_url(body)
            if status in ("failed", "error"):
                raise RuntimeError(f"Token360 video task {task_id} failed: {body}")

            logging.info(f"[Token360] task {task_id} status={status or 'unknown'}; polling again in {poll}s...")
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
        **kwargs,
    ) -> VideoOutput:
        if duration not in ALLOWED_DURATIONS:
            raise ValueError(f"duration must be one of {sorted(ALLOWED_DURATIONS)}; got {duration}")

        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        size = _resolution_to_size(resolution, aspect_ratio)
        annotated_prompt = (
            f"{prompt} --rs {resolution} --rt {aspect_ratio} --dur {duration} --fps {fps}"
        )

        task_id = await self._create_task(
            prompt=annotated_prompt,
            reference_image_paths=reference_image_paths,
            duration=duration,
            size=size,
        )
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


def _resolution_to_size(resolution: str, aspect_ratio: str) -> str:
    h_by_res = {"480p": 480, "720p": 720, "1080p": 1080}
    h = h_by_res.get(resolution, 720)
    try:
        aw, ah = (int(x) for x in aspect_ratio.split(":"))
    except Exception:
        aw, ah = 16, 9
    w = round(h * aw / ah / 8) * 8
    return f"{w}x{h}"
