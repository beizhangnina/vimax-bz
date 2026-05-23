"""Light skin-retouch for Bei via nano-banana-pro i2i.

Removes forehead wrinkles / softens nasolabial lines while preserving
the same face shape, hairstyle, eye/nose/mouth geometry. Saves to
output/bei_retouched.png for use as the new reference photo in the
photoreal chained-video pipeline.
"""
import sys, asyncio, time, os, logging
sys.path.insert(0, "/Users/beizhang/Documents/AI_Dev/vimax-bz")
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

from dotenv import load_dotenv
load_dotenv("/Users/beizhang/Documents/AI_Dev/vimax-bz/.env")
from tools import ImageGeneratorToken360API

OUT = "/Users/beizhang/Documents/AI_Dev/vimax-bz/output/bei_retouched.png"
SRC = "/Users/beizhang/Documents/AI_Dev/vimax-bz/pics/Bei.jpg"

PROMPT = (
    "professional headshot retouch of the same Asian man from the reference photo — "
    "KEEP his exact face shape, identical eyes, identical nose, identical mouth, "
    "identical short black hair with the same natural spikes on top, identical white t-shirt, "
    "same friendly facial expression. ONLY changes: gently smooth skin (remove forehead "
    "wrinkles, soften nasolabial lines, even out skin tone), brighten eyes slightly, "
    "subtle natural look — high-end magazine portrait retouch, NOT plastic, NOT younger, "
    "still recognizably the same person at the same age. Clean studio-quality lighting, "
    "neutral background (light gray), 85mm lens, photorealistic."
)


async def main():
    img = ImageGeneratorToken360API(model="nano-banana-pro")
    t0 = time.monotonic()
    out = await img.generate_single_image(PROMPT, [SRC])
    out.save(OUT)
    print(f"done in {time.monotonic()-t0:.1f}s -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


asyncio.run(main())
