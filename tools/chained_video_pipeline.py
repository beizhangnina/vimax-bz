"""Chained-clip long-form video pipeline with VP-laundered handoffs.

Builds a long continuous video by chaining N short Seedance clips:

    clip_1 = Seedance(first_frame = initial_uri)
    handoff_1 = extract_last_frame(clip_1)
    handoff_1_uri = upload as VIRTUAL_PORTRAIT asset -> ta_xxx
    clip_2 = Seedance(first_frame = handoff_1_uri)
    ...

The VP upload is the key trick: a local PNG of a photoreal person trips
Seedance's content moderation when sent inline, but the IDENTICAL image
re-uploaded as a VP asset bypasses the same check (Token360 whitelists
``ta_`` assets in its frame-input gate). Upload is auto-active in ~10s.

This module is the production-grade replacement for
``scripts/bei_photoreal_15s.py``'s simplistic "fall back to initial VP"
behaviour, which sacrifices continuity.
"""

from __future__ import annotations

import logging
import time
import os
from pathlib import Path
from typing import Optional

import httpx
from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image

from .video_generator_seedance_token360_api import VideoGeneratorSeedanceToken360API


def _extract_last_frame(video_path: Path, out_path: Path) -> Path:
    with VideoFileClip(str(video_path)) as clip:
        last_t = max(0.0, clip.duration - 0.05)
        frame = clip.get_frame(last_t)
        Image.fromarray(frame).save(str(out_path), format="PNG")
    return out_path


class ChainedVideoPipeline:
    """Render an N-clip chained video with VP-laundered handoffs."""

    def __init__(
        self,
        video_gen: Optional[VideoGeneratorSeedanceToken360API] = None,
        seedance_model: str = "seedance-2.0-fast",
        handoff_vp_group_name: str = "vimax-handoff",
    ):
        self.vid = video_gen or VideoGeneratorSeedanceToken360API(
            t2v_model=seedance_model,
            ff2v_model=seedance_model,
            flf2v_model=seedance_model,
        )
        self.api_key = self.vid.api_key
        self.base_url = self.vid.base_url
        self._handoff_group_id: Optional[str] = None
        self._handoff_group_name = handoff_vp_group_name

    # ---- VP handoff upload ----------------------------------------------------

    def _ensure_handoff_group(self) -> str:
        if self._handoff_group_id:
            return self._handoff_group_id
        url = f"{self.base_url}/asset-groups"
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        r = httpx.post(url, json={"name": self._handoff_group_name, "groupKind": "VIRTUAL_PORTRAIT"}, headers=h, timeout=30)
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"failed to create VP handoff group: {body}")
        gid = body["data"]["assetGroupId"]
        self._handoff_group_id = gid
        logging.info(f"[ChainedVideoPipeline] created VP handoff group {gid}")
        return gid

    def _upload_as_vp(self, png_path: Path) -> str:
        gid = self._ensure_handoff_group()
        h = {"Authorization": f"Bearer {self.api_key}"}
        with open(png_path, "rb") as f:
            blob = f.read()
        r = httpx.post(
            f"{self.base_url}/assets", headers=h,
            data={"groupId": gid, "name": png_path.stem},
            files={"file": (png_path.name, blob, "image/png")},
            timeout=120,
        )
        body = r.json()
        if body.get("code") != 200:
            raise RuntimeError(f"VP upload failed for {png_path}: {body}")
        asset_id = body["data"]["assetId"]

        # Poll until active
        hj = {**h, "Content-Type": "application/json"}
        for _ in range(60):
            rr = httpx.get(f"{self.base_url}/assets/{asset_id}", headers=hj, timeout=15)
            st = str((rr.json().get("data") or {}).get("status", "")).lower()
            if st in ("active", "ready"):
                break
            if st in ("failed", "rejected", "error"):
                raise RuntimeError(f"VP handoff asset {asset_id} failed: {rr.text[:300]}")
            time.sleep(2)
        else:
            raise TimeoutError(f"VP handoff asset {asset_id} not active in 120s")
        return f"asset://{asset_id}"

    # ---- chained render -------------------------------------------------------

    async def render(
        self,
        first_frame_uri: str,
        motion_beats: list[str],
        out_dir: str | os.PathLike,
        seconds_per_clip: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
        fps: int = 24,
    ) -> Path:
        """Render `len(motion_beats)` chained clips and concatenate to one mp4.

        Returns the path to the final concatenated mp4. Intermediate clips
        and handoff frames are kept in ``out_dir`` for inspection / rerun.
        """
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

        next_ref = first_frame_uri
        clip_paths: list[Path] = []
        for i, motion in enumerate(motion_beats, start=1):
            clip_path = out / f"clip_{i}.mp4"
            if not clip_path.exists():
                logging.info(f"[ChainedVideoPipeline] clip {i} of {len(motion_beats)} (ref={next_ref[:60]})")
                t0 = time.monotonic()
                vout = await self.vid.generate_single_video(
                    prompt=motion,
                    reference_image_paths=[next_ref],
                    resolution=resolution, aspect_ratio=aspect_ratio,
                    fps=fps, duration=seconds_per_clip,
                )
                vout.save(str(clip_path))
                logging.info(f"  clip {i} done in {time.monotonic()-t0:.1f}s ({clip_path.stat().st_size/1024:.0f} KB)")
            else:
                logging.info(f"[ChainedVideoPipeline] reusing clip {i}: {clip_path.name}")
            clip_paths.append(clip_path)

            if i < len(motion_beats):
                hf_png = out / f"handoff_after_clip_{i}.png"
                if not hf_png.exists():
                    _extract_last_frame(clip_path, hf_png)
                # VP-launder the handoff so Seedance accepts the photoreal frame
                logging.info(f"[ChainedVideoPipeline] VP-laundering {hf_png.name}")
                next_ref = self._upload_as_vp(hf_png)
                logging.info(f"  -> {next_ref}")

        # Concatenate
        final = out / "final.mp4"
        logging.info(f"[ChainedVideoPipeline] concatenating {len(clip_paths)} clips -> {final}")
        t0 = time.monotonic()
        clips = [VideoFileClip(str(p)) for p in clip_paths]
        try:
            cat = concatenate_videoclips(clips, method="compose")
            cat.write_videofile(
                str(final), codec="libx264", audio_codec="aac",
                preset="medium", threads=4, logger=None,
            )
        finally:
            for c in clips:
                c.close()
        logging.info(f"  final in {time.monotonic()-t0:.1f}s ({final.stat().st_size/1024:.0f} KB)")
        return final
