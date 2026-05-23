"""15-second coherent long-shot: Bei in Tokyo rainy neon street.

Pipeline (v2, ChainedVideoPipeline):
  1. Use the retouched Bei photo (output/bei_retouched.png) as new VP asset.
  2. Render 3 × 5s Seedance clips with VP-LAUNDERED handoffs:
       clip_1 first_frame = retouched_vp_uri
       clip_2 first_frame = upload(extract_last(clip_1)) -> new VP URI
       clip_3 first_frame = upload(extract_last(clip_2)) -> new VP URI
     -> each handoff stays in Token360's content-mod whitelist
  3. Concatenate clip_1 + clip_2 + clip_3 -> 15s mp4
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

import httpx
from tools import ChainedVideoPipeline


ROOT = Path("/Users/beizhang/Documents/AI_Dev/vimax-bz")
OUT = ROOT / "output" / "bei_tokyo_neon_15s"
OUT.mkdir(parents=True, exist_ok=True)


# Tokyo Shibuya rainy neon — 3 connected beats (true continuous shot)
MOTION_BEATS = [
    # Beat 1 (0-5s): introduction — lateral tracking shot of him walking
    "the same man is walking across a rain-soaked Shibuya-style crosswalk at night, "
    "wet asphalt reflects giant neon billboards in vivid pink, cyan, and amber; "
    "the camera tracks beside him at chest level (lateral dolly, left-to-right); "
    "rain mist hangs in the neon glow; his white t-shirt and short black spiky hair "
    "catch the colorful highlights; cinematic anamorphic 35mm, shallow depth of field",

    # Beat 2 (5-10s): he stops + turns to camera
    "the man stops mid-crosswalk and slowly turns his head toward the camera; "
    "he holds the gaze with a calm confident expression; rain droplets are subtly "
    "visible in the air; neon ad billboards behind him pulse softly through cyan and "
    "magenta; camera holds steady, very gentle dolly-in (~10% closer); same cinematic "
    "anamorphic 35mm look",

    # Beat 3 (10-15s): close-up push-in, neon reflections in eyes
    "tight medium close-up on the man's face — neon reflections shimmer in his eyes; "
    "he gives a small confident half-smile; ad billboards behind him cycle through "
    "subtle colors and motion; camera pushes in slowly; rain mist diffuses the lights; "
    "shallow depth of field, soft bloom, cinematic anamorphic look",
]


async def main():
    # 1) Wait for retouched photo
    retouched = ROOT / "output" / "bei_retouched.png"
    if not retouched.exists():
        raise SystemExit(f"Missing retouched photo at {retouched}. "
                         "Run scripts/retouch_bei.py first.")
    print(f"\nUsing retouched Bei photo: {retouched}")

    # 2) Upload retouched photo as VIRTUAL_PORTRAIT -> ta_xxx
    key = os.environ["TOKEN360_API_KEY"]
    base = os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1").rstrip("/")
    HJ = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    H = {"Authorization": f"Bearer {key}"}

    print("\nCreating VP group + uploading retouched Bei...")
    r = httpx.post(f"{base}/asset-groups",
                   json={"name": "vimax-bei-retouched", "groupKind": "VIRTUAL_PORTRAIT"},
                   headers=HJ, timeout=30)
    gid = r.json()["data"]["assetGroupId"]
    print(f"  groupId = {gid}")

    with open(retouched, "rb") as f:
        blob = f.read()
    r = httpx.post(f"{base}/assets", headers=H,
                   data={"groupId": gid, "name": "bei-retouched"},
                   files={"file": ("bei_retouched.png", blob, "image/png")},
                   timeout=120)
    asset_id = r.json()["data"]["assetId"]
    print(f"  assetId = {asset_id}")

    # Poll until active
    for i in range(60):
        rr = httpx.get(f"{base}/assets/{asset_id}", headers=HJ, timeout=15)
        st = str((rr.json().get("data") or {}).get("status", "")).lower()
        print(f"  [{i+1}] status={st}")
        if st in ("active", "ready"):
            break
        if st in ("failed", "rejected", "error"):
            raise SystemExit(f"VP upload failed: {rr.text[:300]}")
        time.sleep(2)
    vp_uri = f"asset://{asset_id}"
    print(f"  VP URI = {vp_uri}\n")

    # 3) Render the chained 15s long-shot
    pipe = ChainedVideoPipeline(seedance_model="seedance-2.0-fast",
                                handoff_vp_group_name="vimax-bei-handoff")
    final = await pipe.render(
        first_frame_uri=vp_uri,
        motion_beats=MOTION_BEATS,
        out_dir=OUT,
        seconds_per_clip=5,
        resolution="720p",
        aspect_ratio="16:9",
        fps=24,
    )

    print(f"\n=== BEI TOKYO NEON 15s LONG-SHOT DONE ===")
    print(f"Final: {final}")


asyncio.run(main())
