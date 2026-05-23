"""Demo: Nano Banana Pro i2i + Seedance 2.0 video for Nina (daughter) and YL (friend).

For each subject we:
  1. Pre-process the source photo (resize, ensure reasonable starting size)
  2. Upload to Token360 via the asset uploader (cached by SHA256)
  3. Use Nano Banana Pro i2i to produce a few wholesome scene variations
  4. Pick one image and use Seedance 2.0 (full quality) for a 5s clip

Outputs land under vimax-bz/output/demo_{nina,yl}/

Nina is a minor — scenes are deliberately family-appropriate (school, music,
outdoor activity). YL is a long-time friend, generic portrait / cinematic
scenes.
"""
from __future__ import annotations

import sys
import asyncio
import os
import time
import logging
from pathlib import Path

sys.path.insert(0, "/Users/beizhang/Documents/AI_Dev/vimax-bz")
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

from dotenv import load_dotenv
load_dotenv("/Users/beizhang/Documents/AI_Dev/vimax-bz/.env")

from PIL import Image
from tools import ImageGeneratorToken360API, VideoGeneratorSeedanceToken360API


ROOT = Path("/Users/beizhang/Documents/AI_Dev/vimax-bz")
PICS = ROOT / "pics"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def preprocess(src: Path, dst: Path, max_dim: int = 1024) -> Path:
    """Resize source photo to a sensible max dimension to speed uploads."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_dim:
            s = max_dim / max(w, h)
            im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="JPEG", quality=92, optimize=True)
    return dst


NINA_SCENES = [
    ("classroom",
     "the same little girl from the reference image — same face, same friendly smile, "
     "shoulder-length hair — sitting at a brightly lit elementary school classroom desk, "
     "wearing a clean cotton white t-shirt, focused on a notebook, gentle daylight from "
     "left window, photoreal, eye-level, 35mm"),
    ("piano",
     "the same little girl from the reference image — same face, same hair — sitting at "
     "an upright wooden piano in a warm living room, fingers on the keys, calm focused "
     "expression, soft late-afternoon light, photoreal, 35mm"),
    ("hiking",
     "the same little girl from the reference image — same face, same hair — wearing a "
     "small backpack and standing on a forest hiking trail among tall green trees, looking "
     "back over her shoulder with a bright smile, dappled sunlight, photoreal, 35mm"),
]

YL_SCENES = [
    ("rooftop_sunset",
     "the same woman from the reference image — same face, same long black hair — standing on "
     "a city rooftop at sunset, soft warm golden hour light, distant skyline blurred behind, "
     "wearing the same cream sweater, gentle smile, photoreal cinematic 85mm portrait, "
     "shallow depth of field"),
    ("library_reading",
     "the same woman from the reference image — same face, same long black hair, same cream "
     "sweater — sitting in a quiet wood-panelled library, reading an open book, soft window "
     "light from the side, photoreal editorial, 50mm, calm contemplative mood"),
    ("cafe_window",
     "the same woman from the reference image — same face, same long black hair — by a "
     "minimalist cafe window with a flat white in front of her, looking thoughtfully out "
     "the window, slate-grey daylight, photoreal, 50mm, magazine cover feel"),
]


async def run_subject(
    img_gen: ImageGeneratorToken360API,
    vid_gen: VideoGeneratorSeedanceToken360API,
    subject: str,
    source_photo: Path,
    scenes,
    video_scene_idx: int,
    video_motion: str,
    realface_uri: str | None = None,
):
    """Generate i2i scene variations. Render a Seedance video only when a
    ``realface_uri`` is provided — Seedance's content moderation rejects
    photoreal real-person frames unless the request is bound to a Token360
    RealFace asset (created via the H5 verification flow, see
    scripts/create_realface.py)."""
    sub_out = OUT / f"demo_{subject}"
    sub_out.mkdir(exist_ok=True)

    # Pre-process the source photo to a manageable size before upload.
    base_ref = sub_out / "_source.jpg"
    preprocess(source_photo, base_ref, max_dim=1024)
    print(f"\n=== Subject: {subject} (ref = {base_ref}) ===")

    scene_paths = []
    for label, prompt in scenes:
        t0 = time.monotonic()
        out = await img_gen.generate_single_image(prompt, [str(base_ref)])
        path = sub_out / f"{subject}_{label}.png"
        out.save(str(path))
        size_kb = path.stat().st_size / 1024
        print(f"  ✓ {label:18s} {time.monotonic()-t0:5.1f}s  {size_kb:6.0f} KB  -> {path}")
        scene_paths.append(path)

    if realface_uri is None:
        print(f"\n  (skipping video for {subject}: no RealFace URI — Seedance rejects "
              f"un-verified real-person frames. Run scripts/create_realface.py first.)")
        return

    chosen = scene_paths[video_scene_idx]
    print(f"\n  rendering 5s seedance-2.0-fast video from {chosen.name} "
          f"(bound to RealFace {realface_uri})...")
    t0 = time.monotonic()
    vout = await vid_gen.generate_single_video(
        prompt=video_motion,
        reference_image_paths=[str(chosen)],
        portrait_asset_uris=[realface_uri],
        resolution="720p", aspect_ratio="16:9", fps=24, duration=5,
    )
    vid_path = sub_out / f"{subject}_video.mp4"
    vout.save(str(vid_path))
    size_kb = vid_path.stat().st_size / 1024
    print(f"  ✓ video             {time.monotonic()-t0:5.1f}s  {size_kb:6.0f} KB  -> {vid_path}")


async def main():
    # nano-banana-pro for image (i2i via asset upload).
    # seedance-2.0-fast for video (cheaper + faster than seedance-2.0; sufficient for demo).
    img = ImageGeneratorToken360API(model="nano-banana-pro")
    vid = VideoGeneratorSeedanceToken360API(
        t2v_model="seedance-2.0-fast",
        ff2v_model="seedance-2.0-fast",
        flf2v_model="seedance-2.0-fast",
    )

    # RealFace URIs (optional). Run scripts/create_realface.py for each person
    # and paste the resulting asset://ta_... here.
    NINA_REALFACE = None      # NEVER create a RealFace for a minor.
    YL_REALFACE = None        # Set once YL has done the H5 verification herself.

    await run_subject(
        img, vid, "nina",
        source_photo=PICS / "nina2.JPG",   # the airport one (clearer headshot)
        scenes=NINA_SCENES,
        video_scene_idx=1,                  # piano scene -> video (only if realface_uri set)
        video_motion=(
            "camera slowly pushes in toward the girl playing the piano; "
            "her hands continue moving naturally across the keys; "
            "soft warm interior light"
        ),
        realface_uri=NINA_REALFACE,
    )

    await run_subject(
        img, vid, "yl",
        source_photo=PICS / "YL_resized.jpg",
        scenes=YL_SCENES,
        video_scene_idx=0,                  # rooftop sunset -> video
        video_motion=(
            "camera gentle dolly-in; the woman turns her head a few degrees and "
            "smiles softly; sunset glow on her face; her hair moves slightly in the breeze"
        ),
        realface_uri=YL_REALFACE,
    )

    print("\n=== DEMO COMPLETE ===")
    print(f"Results in {OUT}/demo_nina and {OUT}/demo_yl")


asyncio.run(main())
