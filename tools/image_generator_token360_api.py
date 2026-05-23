"""Image generator backed by Token360's OpenAI-compatible /images/generations.

Plugs into ViMax's ImageGenerator protocol: a class with
async ``generate_single_image(prompt, reference_image_paths, **kwargs) -> ImageOutput``.

KNOWN LIMITATION — reference images:
    Token360's /images/generations is OpenAI-strict (text-prompt only). Sending
    a Seedream-style ``image: [...]`` field is rejected by an AWS WAF (HTTP 403,
    HTML body) — this happens for every model on the gateway, including the
    Seedream / Doubao IDs that DO accept image input on their native providers.

    Behaviour: when ``reference_image_paths`` is passed, we log a warning and
    fall back to text-only generation. This means ViMax's character-consistency
    flow (front-portrait -> side/back via reference) will run, but the rendered
    side/back views won't be conditioned on the front portrait — they'll match
    only via prompt description. For strong consistency, switch the video stage
    to handle character continuity (first-frame I2V with the same portrait), or
    upgrade Token360 to a plan that exposes a reference-aware endpoint.

    To opt into sending references anyway (for future-compat, in case Token360
    relaxes the WAF), instantiate with ``allow_reference_field=True``.

Auth/base_url default from ``TOKEN360_API_KEY`` / ``TOKEN360_BASE_URL`` env vars
when constructor args are empty.
"""

import os
import io
import base64
import logging
from typing import List, Optional

import aiohttp
from PIL import Image
from tenacity import retry, stop_after_attempt

from interfaces.image_output import ImageOutput
from utils.retry import after_func


def _ref_to_b64_url(path: str, max_dim: int = 1024, jpeg_quality: int = 85) -> str:
    """Encode a reference image as a base64 data URL, downsampled to keep
    request bodies under Token360's AWS ELB limit (~1 MB).

    PIL-resize to max_dim on the longest side, then re-encode as JPEG at
    quality=85. A 1024-px photographic image lands ~100-200 KB this way,
    well under the 1 MB ceiling but plenty for character-consistency
    reference at typical generation resolutions.
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


class ImageGeneratorToken360API:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gemini-2.5-flash-image",
        allow_reference_field: bool = False,
        rate_limiter=None,
    ):
        self.api_key = api_key or os.environ.get("TOKEN360_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Token360 API key missing. Pass api_key=... or set TOKEN360_API_KEY in .env."
            )
        self.base_url = (base_url or os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1")).rstrip("/")
        self.endpoint = f"{self.base_url}/images/generations"
        self.model = model
        self.allow_reference_field = allow_reference_field
        self.rate_limiter = rate_limiter

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

        logging.info(f"[Token360] Calling {self.model} to generate image...")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "url",
            "size": size if size is not None else "1024x1024",
            "n": 1,
        }

        if reference_image_paths:
            if self.allow_reference_field:
                payload["image"] = [_ref_to_b64_url(p) for p in reference_image_paths]
                payload["sequential_image_generation"] = "disabled"
            else:
                logging.warning(
                    "[Token360] /images/generations rejects reference images (WAF 403 on the `image` field). "
                    "Ignoring %d reference(s); generation will run from text prompt only. "
                    "Set allow_reference_field=True to opt in anyway.",
                    len(reference_image_paths),
                )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
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
            # Token360 sometimes packs raw base64 into the `url` field
            # (observed with gemini-2.5-flash-image). Detect and route appropriately.
            if val.startswith(("http://", "https://", "data:")):
                return ImageOutput(fmt="url", ext="png", data=val)
            return ImageOutput(fmt="b64", ext="png", data=val)
        raise RuntimeError(f"Token360 image response had no url/b64_json: {response_json}")
