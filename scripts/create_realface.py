"""Create a Token360 portrait asset (REAL_FACE or VIRTUAL_PORTRAIT) for use as
verified-human Seedance video portrait.

** Default: VIRTUAL_PORTRAIT — no H5 verification, just upload your photo. **

The REAL_FACE flow appears broken on Token360 as of 2026-05: the H5 page
shows "Identity authentication successful" but the asset-group status stays
`pending_validation` indefinitely, blocking uploads with
``Asset group is not active`` (HTTP 409). All four hidden activation
endpoints (``/asset-groups/{id}/activate``, ``/validate``, ``/complete``,
``/finalize``, ``/h5/callback``, ``/kyc/sync``...) return 404; PATCH-ing
``status: active`` flips the visible flag but doesn't unlock the internal
validation gate. Existing user accounts have multiple stuck REAL_FACE groups
from prior projects.

Functionally, VIRTUAL_PORTRAIT produces the same ``ta_xxx`` prefix assets
that Seedance accepts as ``frame_images[].image_url.url`` or
``input_references[]``. For self-uploads (you uploading your own photo) the
RealFace anti-impersonation check is redundant.

Usage:
    # Default (recommended): VIRTUAL_PORTRAIT, fully automatic
    uv run python scripts/create_realface.py \\
        --name "Bei Zhang" \\
        --photo pics/Bei.jpg \\
        --out output/bei_realface.txt

    # Optional: try REAL_FACE (will likely stay pending — kept for completeness)
    uv run python scripts/create_realface.py --kind REAL_FACE ...

For REAL_FACE the script:
    1. POST /v1/asset-groups {groupKind: REAL_FACE} -> assetGroupId, h5Link
    2. Prints h5Link as URL + ASCII QR for phone scanning (link valid ~2 min)
    3. Waits for user to complete H5 verification (browser shows
       "Identity authentication successful")
    4. Attempts PATCH status=active (cosmetic only on current backend)
    5. Attempts upload — usually fails with 409 until Token360 fixes their
       activation callback. If it does succeed, polls until asset is active.

For VIRTUAL_PORTRAIT the script:
    1. POST /v1/asset-groups {groupKind: VIRTUAL_PORTRAIT}
    2. POST /v1/assets multipart upload
    3. Poll until status=active (auto, ~10s)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Name of the person (group display name)")
    p.add_argument("--photo", required=True, help="Local image to upload as the reference portrait")
    p.add_argument("--out", default="output/realface_uri.txt", help="File to write the final asset:// URI to")
    p.add_argument("--env", default=".env", help="Path to .env (default: .env)")
    p.add_argument("--kind", default="VIRTUAL_PORTRAIT",
                   choices=("VIRTUAL_PORTRAIT", "REAL_FACE"),
                   help="Asset group kind. VIRTUAL_PORTRAIT (default) skips H5 verification "
                        "and is the recommended path. REAL_FACE adds anti-impersonation H5 "
                        "verification but is currently buggy on Token360's backend.")
    args = p.parse_args()

    load_dotenv(args.env)
    api_key = os.environ.get("TOKEN360_API_KEY")
    if not api_key:
        print("ERROR: TOKEN360_API_KEY missing in env", file=sys.stderr)
        return 2
    base_url = os.environ.get("TOKEN360_BASE_URL", "https://api.token360.ai/v1").rstrip("/")

    if not Path(args.photo).exists():
        print(f"ERROR: photo not found: {args.photo}", file=sys.stderr)
        return 2

    HJ = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    HM = {"Authorization": f"Bearer {api_key}"}

    # --- 1. Create asset group ----------------------------------------------------
    print(f"[1/4] Creating {args.kind} asset group for '{args.name}'...")
    r = httpx.post(
        f"{base_url}/asset-groups",
        json={"name": args.name, "groupKind": args.kind},
        headers=HJ, timeout=30,
    )
    if r.status_code >= 400:
        print(f"  FAILED: HTTP {r.status_code} :: {r.text[:300]}", file=sys.stderr)
        return 1
    body = r.json().get("data") or r.json()
    group_id = body["assetGroupId"]
    h5_link = body.get("h5Link") or body.get("h5_link")
    print(f"  groupId  = {group_id}")

    # --- 2. H5 verification (REAL_FACE only) --------------------------------------
    if args.kind == "REAL_FACE":
        print(f"  h5Link   = {h5_link}")
        print()
        print("[2/4] Open this H5 link on the phone of the person being verified")
        print("      (link expires in ~2 minutes):")
        print()
        print(f"  >>> {h5_link} <<<")
        print()
        try:
            import qrcode  # type: ignore
            qr = qrcode.QRCode(border=1)
            qr.add_data(h5_link)
            qr.make()
            qr.print_ascii(invert=True)
        except Exception:
            print("  (install `qrcode` for an ASCII QR rendering of the link)")

        input("\nPress ENTER once the H5 verification is complete...")

        # Try to PATCH the visible status to active (cosmetic; doesn't unlock
        # the upload gate but keeps the group consistent).
        httpx.patch(
            f"{base_url}/asset-groups/{group_id}",
            json={"status": "active"}, headers=HJ, timeout=15,
        )
    else:
        print("[2/4] (VIRTUAL_PORTRAIT — no H5 verification needed)")

    # --- 3. Upload the portrait file ----------------------------------------------
    print()
    print(f"[3/4] Uploading portrait {args.photo} ...")
    with open(args.photo, "rb") as f:
        blob = f.read()
    files = {"file": (Path(args.photo).name, blob, "image/jpeg")}
    data = {"groupId": group_id, "name": Path(args.photo).stem}
    r = httpx.post(f"{base_url}/assets", headers=HM, data=data, files=files, timeout=120)
    if r.status_code >= 400:
        print(f"  upload FAILED: HTTP {r.status_code} :: {r.text[:300]}", file=sys.stderr)
        return 1
    asset = r.json().get("data") or r.json()
    asset_id = asset["assetId"]
    print(f"  assetId  = {asset_id}")

    # --- 4. Poll until active -----------------------------------------------------
    print()
    print("[4/4] Waiting for asset to become active (this can be a few seconds)...")
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        r = httpx.get(f"{base_url}/assets/{asset_id}", headers=HJ, timeout=30)
        status = str((r.json().get("data") or r.json()).get("status", "")).lower()
        print(f"  status={status}")
        if status in ("active", "ready"):
            break
        if status in ("failed", "error", "rejected"):
            print(f"  asset FAILED: {r.text[:300]}", file=sys.stderr)
            return 1
        time.sleep(2)
    else:
        print("  TIMEOUT waiting for asset", file=sys.stderr)
        return 1

    asset_uri = f"asset://{asset_id}"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(asset_uri + "\n")

    print()
    print("================================================================")
    print(f"  RealFace asset ready:  {asset_uri}")
    print(f"  Saved to:              {out_path}")
    print()
    print("  Use this URI in video generation by passing:")
    print(f"      portrait_asset_uris=['{asset_uri}']")
    print("  to VideoGeneratorSeedanceToken360API.generate_single_video(...)")
    print("================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
