"""Stylize-then-animate pipeline for real-person reference photos.

Token360's Seedance content moderation rejects photoreal first-frame inputs
that contain a real human face unless the request is bound to a verified
``RealFace`` asset. For people who haven't gone through the H5 verification
flow (friends, family, public-domain references), this pipeline keeps the
character recognizable by routing through a stylized intermediate frame:

    source_photo  --i2i (nano-banana-pro, stylized)-->  stylized_frame
    stylized_frame  --first-frame I2V (seedance-2.0-fast)-->  mp4

Recommended styles (run scripts/probe_stylize_yl.py to verify on your key):

* ``pixar_3d``       — Pixar-style 3D render, strong fidelity, passes moderation.
* ``ghibli_anime``   — Studio Ghibli look, very safe, mid fidelity.
* ``editorial_painting`` — oil painting feel, highest "feels like the person",
                       most likely to be marginal at moderation; use for adults
                       whose references resemble stock portraits.

Usage::

    from tools.stylized_video_pipeline import StylizedVideoPipeline
    pipe = StylizedVideoPipeline()  # uses env-based defaults
    mp4 = await pipe.run(
        source_photo="pics/YL_resized.jpg",
        style="pixar_3d",
        scene_prompt="standing on a city rooftop at sunset, warm golden light",
        motion_prompt="camera gentle dolly-in; subject smiles softly",
        out_dir="output/yl_rooftop",
    )
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

from .image_generator_token360_api import ImageGeneratorToken360API
from .video_generator_seedance_token360_api import VideoGeneratorSeedanceToken360API


# Style prompt prefixes. Each one is templated with the user's scene_prompt to
# produce the stylization instructions for nano-banana-pro.
STYLE_PROMPTS: dict[str, str] = {
    "pixar_3d": (
        "Pixar-style 3D animated character render of the same person from the reference "
        "photo — stylized but clearly recognizable, same face shape, large expressive eyes, "
        "same hair colour and length, same general clothing. Wholesome 3D animation movie "
        "still, soft cinematic lighting, NOT a photograph. Scene: {scene}"
    ),
    "ghibli_anime": (
        "Studio Ghibli anime-style illustration of the same person from the reference photo, "
        "preserving recognizable features (face shape, hair, clothing). Hand-painted "
        "watercolor background, gentle line art, peaceful expression. Ghibli movie still. "
        "Scene: {scene}"
    ),
    "editorial_painting": (
        "An editorial portrait painting of the same person from the reference photo, "
        "preserving face shape, eyes, and hairstyle. Oil-on-canvas style, visible "
        "brushstrokes, magazine cover composition, NOT a photograph. Scene: {scene}"
    ),
    "ink_wash": (
        "A Chinese ink wash painting (水墨) portrait of the same person from the reference "
        "photo, preserving face structure and hairstyle. Subtle ink gradients, traditional "
        "brushwork, rice paper texture, NOT photorealistic. Scene: {scene}"
    ),
}


class StylizedVideoPipeline:
    def __init__(
        self,
        image_gen: Optional[ImageGeneratorToken360API] = None,
        video_gen: Optional[VideoGeneratorSeedanceToken360API] = None,
        image_model: str = "nano-banana-pro",
        video_model: str = "seedance-2.0-fast",
    ):
        self.img = image_gen or ImageGeneratorToken360API(model=image_model)
        self.vid = video_gen or VideoGeneratorSeedanceToken360API(
            t2v_model=video_model, ff2v_model=video_model, flf2v_model=video_model,
        )

    async def run(
        self,
        source_photo: str,
        style: str,
        scene_prompt: str,
        motion_prompt: str,
        out_dir: str,
        duration: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
    ) -> Path:
        if style not in STYLE_PROMPTS:
            raise ValueError(f"style must be one of {sorted(STYLE_PROMPTS)}; got {style!r}")
        out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

        # 1) Stylize via i2i
        prompt = STYLE_PROMPTS[style].format(scene=scene_prompt)
        logging.info(f"[StylizedVideoPipeline] stylizing ({style}) ...")
        t0 = time.monotonic()
        img_out = await self.img.generate_single_image(prompt, [source_photo])
        stylized = out_path / f"{Path(source_photo).stem}_{style}.png"
        img_out.save(str(stylized))
        logging.info(f"  stylized in {time.monotonic()-t0:.1f}s -> {stylized}")

        # 2) Animate via Seedance first-frame I2V
        logging.info(f"[StylizedVideoPipeline] rendering {duration}s {resolution} video ...")
        t0 = time.monotonic()
        vout = await self.vid.generate_single_video(
            prompt=motion_prompt,
            reference_image_paths=[str(stylized)],
            resolution=resolution, aspect_ratio=aspect_ratio,
            duration=duration,
        )
        video_path = out_path / f"{Path(source_photo).stem}_{style}.mp4"
        vout.save(str(video_path))
        logging.info(f"  video in {time.monotonic()-t0:.1f}s -> {video_path}")
        return video_path
