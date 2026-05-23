"""15-second cinematic Bei video via Pixar-3D stylization + chained Seedance clips.

Pipeline:
  1. Stylize pics/Bei.jpg -> Pixar 3D cinematic urban-rooftop first frame
     (passes Seedance content moderation; 'real person' detector silenced)
  2. Render clip 1 (5s seedance-2.0-fast) from that first frame
  3. Extract last frame of clip 1 -> first frame of clip 2
  4. Render clip 2 (5s) from extracted frame
  5. Repeat for clip 3
  6. Concatenate clip1 + clip2 + clip3 = 15s mp4

Output: output/bei_15s/{bei_pixar_frame.png, clip_{1,2,3}.mp4, bei_15s_final.mp4}
"""
from __future__ import annotations

import sys
import asyncio
import time
import os
import logging
from pathlib import Path

sys.path.insert(0, "/Users/beizhang/Documents/AI_Dev/vimax-bz")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

from dotenv import load_dotenv
load_dotenv("/Users/beizhang/Documents/AI_Dev/vimax-bz/.env")

from moviepy import VideoFileClip, concatenate_videoclips
from PIL import Image
from tools import ImageGeneratorToken360API, VideoGeneratorSeedanceToken360API


ROOT = Path("/Users/beizhang/Documents/AI_Dev/vimax-bz")
PICS = ROOT / "pics"
OUT = ROOT / "output" / "bei_15s"
OUT.mkdir(parents=True, exist_ok=True)


# --- Pixar 3D stylization with cinematic context -----------------------------
PIXAR_FIRST_FRAME = (
    "Pixar-style 3D animated character render of the same man from the reference photo — "
    "stylized but clearly recognizable: same face shape, same short black hair with the "
    "natural spikes on top, same general build, wearing a clean white t-shirt. "
    "Standing on a city rooftop at dusk, blurred neon skyline behind him with warm orange "
    "and cool blue lights, gentle wind catches his sleeves; he is in the foreground, slightly "
    "off-center to the right, looking thoughtfully into the distance. "
    "Cinematic Pixar movie still, soft directional light, depth of field, 3D animation, "
    "NOT a photograph."
)


# Three connected motion beats, each ~5s. Each starts from the previous clip's last frame.
MOTION_BEATS = [
    "camera holds steady at chest level; the man slowly turns his head toward the camera; "
    "his sleeves and hair move slightly in the breeze; dusk light shifts subtly on his face",
    "the man takes a single step forward, his expression softens into a small confident smile; "
    "camera does a gentle dolly-in (~20% closer); rooftop lights start to twinkle behind him",
    "the man looks up briefly toward the sky, then back at the camera with a calm settled "
    "expression; camera pulls back smoothly to a slightly wider framing as warm city lights "
    "intensify behind him",
]


def extract_last_frame(video_path: Path, out_path: Path) -> Path:
    """Extract the final frame of a video as a PNG (for use as next first_frame)."""
    with VideoFileClip(str(video_path)) as clip:
        last_t = max(0.0, clip.duration - 0.05)
        frame = clip.get_frame(last_t)
        Image.fromarray(frame).save(str(out_path), format="PNG")
    return out_path


async def main():
    img = ImageGeneratorToken360API(model="nano-banana-pro")
    vid = VideoGeneratorSeedanceToken360API(
        t2v_model="seedance-2.0-fast",
        ff2v_model="seedance-2.0-fast",
        flf2v_model="seedance-2.0-fast",
    )

    # 1) Stylize Bei
    first_frame_path = OUT / "bei_pixar_frame.png"
    if not first_frame_path.exists():
        print("\n[1/5] stylizing Bei.jpg -> Pixar 3D cinematic rooftop ...")
        t0 = time.monotonic()
        out = await img.generate_single_image(PIXAR_FIRST_FRAME, [str(PICS / "Bei.jpg")])
        out.save(str(first_frame_path))
        print(f"  done in {time.monotonic()-t0:.1f}s -> {first_frame_path} "
              f"({first_frame_path.stat().st_size/1024:.0f} KB)")
    else:
        print(f"\n[1/5] reusing existing first frame: {first_frame_path}")

    # 2-4) Render 3 chained 5s clips
    next_frame = first_frame_path
    clip_paths: list[Path] = []
    for i, motion in enumerate(MOTION_BEATS, start=1):
        clip_path = OUT / f"clip_{i}.mp4"
        if clip_path.exists():
            print(f"\n[{i+1}/5] reusing existing clip_{i}.mp4")
        else:
            print(f"\n[{i+1}/5] rendering clip {i} from {next_frame.name} ...")
            t0 = time.monotonic()
            vout = await vid.generate_single_video(
                prompt=motion,
                reference_image_paths=[str(next_frame)],
                resolution="720p", aspect_ratio="16:9", fps=24, duration=5,
            )
            vout.save(str(clip_path))
            print(f"  clip {i} done in {time.monotonic()-t0:.1f}s "
                  f"({clip_path.stat().st_size/1024:.0f} KB) -> {clip_path}")
        clip_paths.append(clip_path)

        # Extract last frame for next clip (unless this is the last clip)
        if i < len(MOTION_BEATS):
            next_frame = OUT / f"handoff_after_clip_{i}.png"
            if not next_frame.exists():
                extract_last_frame(clip_path, next_frame)
                print(f"  extracted last frame -> {next_frame}")

    # 5) Concatenate clips into a single 15s mp4
    final_path = OUT / "bei_15s_final.mp4"
    print(f"\n[5/5] concatenating {len(clip_paths)} clips -> {final_path} ...")
    t0 = time.monotonic()
    clips = [VideoFileClip(str(p)) for p in clip_paths]
    try:
        concat = concatenate_videoclips(clips, method="compose")
        concat.write_videofile(
            str(final_path),
            codec="libx264",
            audio_codec="aac",
            preset="medium",
            threads=4,
            logger=None,
        )
    finally:
        for c in clips:
            c.close()
    print(f"  final done in {time.monotonic()-t0:.1f}s "
          f"({final_path.stat().st_size/1024:.0f} KB) -> {final_path}")
    print("\n=== BEI 15s CINEMATIC DONE ===")


asyncio.run(main())
