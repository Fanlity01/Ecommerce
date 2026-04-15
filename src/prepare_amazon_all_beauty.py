import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Amazon Reviews 2023 All_Beauty for the multimodal project")
    parser.add_argument("--out_dir", type=str, default="exampless/amazon_all_beauty")
    parser.add_argument("--max_reviews", type=int, default=20000, help="Maximum number of reviews to export")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before truncating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_images", action="store_true", help="Download product images locally")
    parser.add_argument("--image_timeout", type=int, default=20)
    return parser.parse_args()


def sentiment_label_from_rating(rating: float) -> int:
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def credibility_label_heuristic(
    text: str,
    rating: float,
    helpful_vote: int,
    verified_purchase: bool,
    user_review_count: int,
    user_account_days: int,
) -> int:
    text = (text or "").strip()
    length = len(text)
    exclam = text.count("!") + text.count("！")
    extreme = rating in (1, 5)

    suspicious = (
        (not verified_purchase and helpful_vote == 0 and length < 20)
        or (user_review_count <= 1 and user_account_days <= 7 and helpful_vote == 0)
        or (extreme and exclam >= 3 and length < 30)
    )
    return 0 if suspicious else 1


def pick_first_url(images_field: Any) -> Optional[str]:
    if not images_field:
        return None

    # Hugging Face metadata style: {"hi_res": [...], "large": [...], "thumb": [...], ...}
    if isinstance(images_field, dict):
        for key in ("hi_res", "large", "thumb"):
            value = images_field.get(key)
            if isinstance(value, list):
                for x in value:
                    if isinstance(x, str) and x.strip():
                        return x.strip()
            elif isinstance(value, str) and value.strip():
                return value.strip()

    # Native-json style or review-image style: list[dict] / list[str]
    if isinstance(images_field, list):
        for item in images_field:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for key in ("hi_res", "large", "thumb", "image", "url"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        return None

    if isinstance(images_field, str) and images_field.strip():
        return images_field.strip()

    return None


def download_image(url: str, image_dir: Path, stem: str, timeout: int = 20) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    file_path = image_dir / f"{stem}{suffix}"
    if file_path.exists():
        return file_path.name

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    file_path.write_bytes(r.content)
    return file_path.name


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Loading review dataset...")
    review_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True,
    )

    print("Loading metadata dataset...")
    meta_ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_All_Beauty",
        split="full",
        trust_remote_code=True,
    )

    if args.shuffle:
        review_ds = review_ds.shuffle(seed=args.seed)

    if args.max_reviews and args.max_reviews < len(review_ds):
        review_ds = review_ds.select(range(args.max_reviews))

    print("Building product image map...")
    meta_image_map: Dict[str, Optional[str]] = {}
    for row in tqdm(meta_ds, total=len(meta_ds), desc="meta"):
        parent_asin = str(row.get("parent_asin", "")).strip()
        if not parent_asin:
            continue
        meta_image_map[parent_asin] = pick_first_url(row.get("images"))

    review_rows = []
    cred_rows = []

    user_count: Dict[str, int] = {}
    user_first_ts: Dict[str, int] = {}

    print("Scanning review statistics...")
    review_list = []
    for row in tqdm(review_ds, total=len(review_ds), desc="reviews-pass1"):
        review_list.append(row)
        user_id = str(row.get("user_id", "unknown_user"))
        ts = int(row.get("timestamp", 0) or 0)
        user_count[user_id] = user_count.get(user_id, 0) + 1
        if user_id not in user_first_ts or ts < user_first_ts[user_id]:
            user_first_ts[user_id] = ts

    print("Creating CSV rows...")
    for idx, row in enumerate(tqdm(review_list, total=len(review_list), desc="reviews-pass2")):
        text = (str(row.get("title", "")) + " " + str(row.get("text", ""))).strip()
        if not text:
            continue

        rating = float(row.get("rating", 3.0) or 3.0)
        parent_asin = str(row.get("parent_asin", row.get("asin", ""))).strip()
        user_id = str(row.get("user_id", "unknown_user"))
        helpful_vote = int(row.get("helpful_vote", 0) or 0)
        verified_purchase = bool(row.get("verified_purchase", False))
        timestamp_ms = int(row.get("timestamp", 0) or 0)

        review_image_url = pick_first_url(row.get("images"))
        image_url = review_image_url or meta_image_map.get(parent_asin)

        review_key = f"{user_id}_{parent_asin}_{timestamp_ms}_{idx}"
        review_id = hashlib.md5(review_key.encode("utf-8")).hexdigest()[:16]

        image_path = ""
        if image_url and args.download_images:
            try:
                image_path = download_image(image_url, image_dir, review_id, timeout=args.image_timeout)
            except Exception:
                image_path = ""

        ts = pd.to_datetime(timestamp_ms, unit="ms", errors="coerce")
        if pd.isna(ts):
            ts_str = "2026-01-01 00:00:00"
            user_account_days = 1
        else:
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            first_ts = pd.to_datetime(user_first_ts.get(user_id, timestamp_ms), unit="ms", errors="coerce")
            if pd.isna(first_ts):
                user_account_days = 1
            else:
                user_account_days = max((ts - first_ts).days + 1, 1)

        senti_label = sentiment_label_from_rating(rating)
        cred_label = credibility_label_heuristic(
            text=text,
            rating=rating,
            helpful_vote=helpful_vote,
            verified_purchase=verified_purchase,
            user_review_count=user_count.get(user_id, 1),
            user_account_days=user_account_days,
        )

        review_rows.append(
            {
                "review_id": review_id,
                "text": text,
                "image_path": image_path,
                "label": senti_label,
            }
        )

        cred_rows.append(
            {
                "review_id": review_id,
                "text": text,
                "image_path": image_path,
                "rating": rating,
                "timestamp": ts_str,
                "user_id": user_id,
                "user_review_count": user_count.get(user_id, 1),
                "user_account_days": user_account_days,
                "helpful_votes": helpful_vote,
                "verified_purchase": int(verified_purchase),
                "label": cred_label,
            }
        )

    sentiment_df = pd.DataFrame(review_rows)
    credibility_df = pd.DataFrame(cred_rows)

    sentiment_csv = out_dir / "sentiment_train.csv"
    credibility_csv = out_dir / "credibility_train.csv"
    sentiment_df.to_csv(sentiment_csv, index=False, encoding="utf-8-sig")
    credibility_df.to_csv(credibility_csv, index=False, encoding="utf-8-sig")

    print(f"Saved sentiment csv: {sentiment_csv}")
    print(f"Saved credibility csv: {credibility_csv}")
    print(f"Saved image dir: {image_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
