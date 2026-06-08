# vision/image_ingestor.py

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from vision.clip_engine import embed_image_from_url
from vision.vector_store import upsert_vehicle

load_dotenv()

INPUT_FILE = "outputs/scored_dataset.csv"


def extract_first_image_url(raw_url_string: str) -> str | None:
    """
    Parses comma-separated URL string, returns first valid image URL.
    Skips video files (.mp4, .mov, .avi).
    """
    if not raw_url_string or str(raw_url_string).strip() == "nan":
        return None

    urls = [u.strip() for u in str(raw_url_string).split(",")]

    for url in urls:
        # Skip videos
        if any(url.lower().endswith(ext) for ext in [".mp4", ".mov", ".avi", ".webm"]):
            continue
        # Accept image URLs
        if any(ext in url.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]) or \
           "gumlet.io" in url or "clearquote" in url:
            return url

    return None


def run_ingestion(
    tenant_id: str = "spinny_india_prod",
    sample_size: int = None,
    verbose: bool = True
) -> dict:

    if verbose:
        print(f"\n{'='*60}")
        print("  VisionLens — Image Ingestion Pipeline")
        print(f"{'='*60}")

    df = pd.read_csv(INPUT_FILE)

    if sample_size:
        df = df.head(sample_size)
        if verbose:
            print(f"\n  Sample mode: {sample_size} rows")
    else:
        if verbose:
            print(f"\n  Full dataset: {len(df):,} rows")

    stats = {"total": len(df), "success": 0, "failed": 0, "skipped": 0}

    if verbose:
        print(f"  Tenant: {tenant_id}\n  {'─'*56}")

    for i, row in tqdm(df.iterrows(), total=len(df), disable=not verbose):
        # Parse first valid image URL from comma-separated list
        raw_url = str(row.get("media_url", ""))
        url = extract_first_image_url(raw_url)

        if not url:
            stats["skipped"] += 1
            continue

        # Generate CLIP embedding
        vector = embed_image_from_url(url)

        if vector is None:
            stats["failed"] += 1
            continue

        # Get city from URL or default
        city = str(row.get("city", "Unknown")) if "city" in row.index else "Unknown"

        success = upsert_vehicle(
            vehicle_id=str(row.get("lead_id", f"vehicle_{i}")),
            tenant_id=tenant_id,
            image_url=url,
            vector=vector,
            panel=str(row.get("panel", "")),
            damage_score=float(row.get("damage_score", 0.0)),
            cost_bucket=str(row.get("cost_bucket", "")),
            price=int(row.get("cost_min", 1000)),
            city=city,
            img_index=0
        )

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

        if verbose and stats["success"] % 50 == 0 and stats["success"] > 0:
            print(f"\n  ✅ {stats['success']} images indexed...")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Ingestion Complete")
        print(f"  {'─'*56}")
        print(f"  Total    : {stats['total']:,}")
        print(f"  Indexed  : {stats['success']:,}")
        print(f"  Failed   : {stats['failed']:,}")
        print(f"  Skipped  : {stats['skipped']:,}")
        print(f"{'='*60}\n")

    return stats


if __name__ == "__main__":
    run_ingestion(
        tenant_id="spinny_india_prod",
        sample_size=10000,
        verbose=True
    )