"""Probe Token360 to find working chat, image, and video models.

Run: `uv run python scripts/probe_token360.py`

Reads TOKEN360_API_KEY / TOKEN360_BASE_URL from .env (project root).
Prints lists of working chat/image models, and existing video model IDs
(video gen uses async tasks, not chat completions; we treat any non-503/404
as "model exists, gateway accepts the ID").
"""
import os
import sys
import json

import httpx
from openai import OpenAI
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(ROOT, ".env"))

API_KEY = os.environ["TOKEN360_API_KEY"]
BASE_URL = os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def classify(err: Exception) -> str:
    msg = str(err)
    if "503" in msg and "No available provider" in msg:
        return "503 no provider"
    if "404" in msg:
        return "404 not found"
    if "401" in msg:
        return "401 unauthorized"
    if "402" in msg or "billing" in msg.lower() or "quota" in msg.lower():
        return "402 billing/quota"
    if "400" in msg:
        return "400 (model exists, wrong endpoint/params)"
    return msg[:120]


def probe_chat(model: str) -> str:
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2,
        )
        return "OK"
    except Exception as e:
        return classify(e)


def probe_image(model: str) -> str:
    try:
        client.images.generate(model=model, prompt="a red apple", n=1, size="1024x1024")
        return "OK"
    except Exception as e:
        return classify(e)


def probe_video_via_chat(model: str) -> str:
    """Use chat completions to confirm gateway recognises the model ID.
    A 400 here means 'model exists but chat is the wrong endpoint' - good enough."""
    return probe_chat(model)


def probe_video_native(model: str) -> str:
    """Try the native /videos endpoint with a minimal payload."""
    try:
        r = httpx.post(
            f"{BASE_URL}/videos",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "prompt": "a cat"},
            timeout=30,
        )
        if r.status_code == 200:
            return f"OK 200: {json.dumps(r.json())[:120]}"
        return f"{r.status_code}: {r.text[:120]}"
    except Exception as e:
        return f"ERR {str(e)[:120]}"


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


# ---------------------------------------------------------------------------
section("Full catalog from GET /models")
try:
    catalog = client.models.list()
    all_ids = sorted(m.id for m in catalog.data)
    print(f"Token360 advertises {len(all_ids)} model IDs:")
    for mid in all_ids:
        print(f"  {mid}")
except Exception as e:
    print(f"FAIL listing models: {classify(e)}")
    all_ids = []

# ---------------------------------------------------------------------------
section("Chat / LLM candidates")
llm_candidates = [
    # OpenAI
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini",
    # Anthropic (via Token360)
    "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-3-5-sonnet-20241022",
    "claude-haiku-4-5-20251001", "claude-3-5-haiku-20241022",
    # Google
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    # Chinese providers
    "qwen-max", "qwen-plus", "qwen2.5-72b-instruct",
    "deepseek-v3", "deepseek-chat", "deepseek-r1",
    "doubao-1-5-pro-32k", "doubao-1-5-pro-256k",
    "glm-4-plus", "moonshot-v1-32k",
]
llm_ok = []
for m in llm_candidates:
    status = probe_chat(m)
    flag = "OK  " if status == "OK" else "FAIL"
    print(f"  {flag} {m:40s} {status if status != 'OK' else ''}")
    if status == "OK":
        llm_ok.append(m)

# ---------------------------------------------------------------------------
section("Image-gen candidates (POST /images/generations)")
image_candidates = [
    "dall-e-3", "dall-e-2", "gpt-image-1",
    "doubao-seedream-4-0-250828", "doubao-seedream-3-0",
    "seedream-4.0", "seedream-3.0",
    "nano-banana", "nano-banana-pro", "nanobanana", "nanobanana-pro",
    "gemini-2.5-flash-image", "gemini-2.5-flash-image-preview",
    "flux-1.1-pro", "flux-kontext-pro", "stable-diffusion-3.5-large",
]
image_ok = []
for m in image_candidates:
    status = probe_image(m)
    flag = "OK  " if status == "OK" else "FAIL"
    print(f"  {flag} {m:40s} {status if status != 'OK' else ''}")
    if status == "OK":
        image_ok.append(m)

# ---------------------------------------------------------------------------
section("Video-gen candidate IDs (chat-endpoint probe; 400 = exists)")
video_candidates = [
    # Seedance / Dreamina (best on Token360 per UI)
    "dreamina-seedance-2.0", "dreamina-seedance-2.0-pro", "dreamina-seedance-2.0-fast",
    "seedance-2.0-pro", "seedance-2.0", "seedance-2.0-fast", "seedance-2.0-lite",
    "seedance-1.5-pro", "seedance-1.0-pro", "seedance-1.0",
    "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-lite-i2v-250428",
    # Google Veo
    "veo-3", "veo-2", "veo-3.0-generate-001",
    # MiniMax / Hailuo (also on Token360 per UI)
    "minimax-hailuo-2.3", "minimax-hailuo-02", "hailuo-02", "minimax-video-01",
    # Kling
    "kling-v2.1", "kling-v2", "kling-v1.6",
    # OpenAI Sora
    "sora-2", "sora-2-pro",
    # Wan / Runway
    "wan-2.1", "wan-2.2",
    "runway-gen-4", "runway-gen-3-alpha",
]
video_exists = []
for m in video_candidates:
    status = probe_video_via_chat(m)
    exists = status == "OK" or status.startswith("400")
    flag = "EXISTS" if exists else "FAIL  "
    print(f"  {flag} {m:40s} {status}")
    if exists:
        video_exists.append(m)

# ---------------------------------------------------------------------------
section("Video-gen native endpoint probe (POST /videos)")
for m in video_exists[:5]:  # only top candidates to save quota
    print(f"  {m}: {probe_video_native(m)}")

# ---------------------------------------------------------------------------
print()
print("WORKING chat models:", llm_ok)
print("WORKING image models:", image_ok)
print("EXISTING video model IDs:", video_exists)

if not video_exists:
    print()
    print("WARN: no video model IDs recognised. Token360 may use a different naming "
          "convention or video endpoint. Check https://www.token360.ai/en-US/docs.")
    sys.exit(1)
