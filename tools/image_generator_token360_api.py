"""Image generator backed by Token360's /images/generations.

Default model is Nano Banana Pro (``nano-banana-pro``) — Google's reference-image
aware image generator. Reference images are uploaded to Token360's Assets API
and passed back into the request as ``asset://`` URIs, which is the only shape
that survives the gateway's AWS WAF (inline base64 reference images are blocked
above a few KB).

Pipe::

    POST /v1/asset-groups             # one-time, cached on disk
    POST /v1/assets   (multipart)     # per unique file, cached by sha256
    GET  /v1/assets/{id}              # poll status -> active
    POST /v1/images/generations       # body uses reference_images: ["asset://..."]

Plugs into ViMax's ImageGenerator protocol via the async
``generate_single_image(prompt, reference_image_paths, **kwargs) -> ImageOutput``
method. Set ``api_key`` + ``base_url`` explicitly or rely on the
``TOKEN360_API_KEY`` / ``TOKEN360_BASE_URL`` env vars.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional

import aiohttp
from tenacity import retry, stop_after_attempt

from interfaces.image_output import ImageOutput
from utils.retry import after_func

from .token360_assets import Token360AssetUploader


class ImageGeneratorToken360API:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "nano-banana-pro",
        reference_field: str = "reference_images",
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
        self.endpoint = f"{self.base_url}/images/generations"
        self.model = model
        self.reference_field = reference_field
        self.rate_limiter = rate_limiter
        self._uploader = Token360AssetUploader(api_key=self.api_key, base_url=self.base_url)

    @retry(stop=stop_after_attempt(3), after=after_func)
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        size: Optional[str] = None,
        **kwargs,
    ) -> ImageOutput:
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "url",
            "size": size if size is not None else "1024x1024",
            "n": 1,
        }

        if reference_image_paths:
            asset_uris = [await self._uploader.upload(p) for p in reference_image_paths]
            payload[self.reference_field] = asset_uris
            logging.info(
                f"[Token360] {self.model} i2i with {len(asset_uris)} reference(s)"
            )
        else:
            logging.info(f"[Token360] {self.model} text-to-image")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as response:
                response_json = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"Token360 image gen failed ({response.status}): {response_json}"
                    )

        entry = response_json["data"][0]
        if "b64_json" in entry and entry["b64_json"]:
            return ImageOutput(fmt="b64", ext="png", data=entry["b64_json"])
        if "url" in entry and entry["url"]:
            val = entry["url"]
            # Some models pack raw base64 into `url`; treat anything not
            # starting with a real scheme as base64 payload.
            if val.startswith(("http://", "https://", "data:")):
                return ImageOutput(fmt="url", ext="png", data=val)
            return ImageOutput(fmt="b64", ext="png", data=val)
        raise RuntimeError(f"Token360 image response had no url/b64_json: {response_json}")
