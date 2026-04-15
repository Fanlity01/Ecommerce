import argparse
import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


SUPPORTED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".json.gz",
    ".jsonl.gz",
    ".parquet",
    ".csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local Amazon Reviews 2023 All_Beauty files into training CSVs"
    )
    parser.add_argument("--review_file", type=str, required=True, help="Local review file path")
    parser.add_argument("--meta_file", type=str, default=None, help="Local metadata file path (optional)")
    parser.add_argument("--out_dir", type=str, default="exampless/amazon_all_beauty")
    parser.add_argument("--max_reviews", type=int, default=20000, help="Maximum reviews to export after optional shuffle")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selected reviews before truncating")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_images", action="store_true", help="Download one image per review/product")
    parser.add_argument("--image_timeout", type=int, default=20)
    parser.add_argument("--min_text_len", type=int, default=3, help="Drop reviews with too-short text")
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


def file_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".jsonl.gz"):
        return ".jsonl.gz"
    if name.endswith(".json.gz"):
        return ".json.gz"
    return path.suffix.lower()


def open_text_auto(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def normalize_record(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(row)


def iter_json_lines(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text_auto(path) as f:
        first_non_ws = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        if first_non_ws is None:
            return
        f.seek(0)
        if first_non_ws == "[":
            data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    yield normalize_record(row)
            else:
                raise ValueError(f"Expected a JSON array in {path}")
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield normalize_record(json.loads(line))


def iter_parquet(path: Path) -> Iterator[Dict[str, Any]]:
    df = pd.read_parquet(path)
    for row in df.to_dict(orient="records"):
        yield normalize_record(row)


def iter_csv(path: Path) -> Iterator[Dict[str, Any]]:
    df = pd.read_csv(path)
    for row in df.to_dict(orient="records"):
        yield normalize_record(row)


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    suffix = file_suffix(p)
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {p.name}. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    if suffix in {".json", ".jsonl", ".json.gz", ".jsonl.gz"}:
        return list(iter_json_lines(p))
    if suffix == ".parquet":
        return list(iter_parquet(p))
    if suffix == ".csv":
        return list(iter_csv(p))
    raise ValueError(f"Unsupported file type: {p}")


def pick_first_url(images_field: Any) -> Optional[str]:
    if images_field is None or images_field == "":
        return None

    if isinstance(images_field, str):
        s = images_field.strip()
        if not s:
            return None
        if s.startswith("http://") or s.startswith("https://"):
            return s
        try:
            parsed = json.loads(s)
            return pick_first_url(parsed)
        except Exception:
            return None

    if isinstance(images_field, dict):
        for key in ("hi_res", "large", "thumb", "images", "image", "url"):
            value = images_field.get(key)
            if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                return value.strip()
            if isinstance(value, list):
                url = pick_first_url(value)
                if url:
                    return url
        return None

    if isinstance(images_field, list):
        for item in images_field:
            url = pick_first_url(item)
            if url:
                return url
        return None

    return None


def download_image(url: str, image_dir: Path, stem: str, timeout: int = 20) -> str:
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    file_path = image_dir / f"{stem}{suffix}"
    if file_path.exists():
        return file_path.name

    headers = {
        "User-Agent": "Mozilla/5.0",
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    file_path.write_bytes(response.content)
    return file_path.name


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return str(value)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except Exception:
        return default


TRUE_STRINGS = {"true", "1", "yes", "y", "t"}


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in TRUE_STRINGS
    return default


def coalesce(*values: Any) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def normalize_timestamp(raw_value: Any) -> tuple[str, int]:
    if raw_value is None or raw_value == "":
        return "2026-01-01 00:00:00", 0

    if isinstance(raw_value, str):
        raw = raw_value.strip()
        if raw.isdigit():
            ts_num = int(raw)
        else:
            dt = pd.to_datetime(raw, errors="coerce")
            if pd.isna(dt):
                return "2026-01-01 00:00:00", 0
            return dt.strftime("%Y-%m-%d %H:%M:%S"), int(dt.timestamp() * 1000)
    else:
        ts_num = safe_int(raw_value, 0)

    if ts_num <= 0:
        return "2026-01-01 00:00:00", 0

    # Heuristic: seconds vs milliseconds vs microseconds
    if ts_num < 10**11:
        dt = pd.to_datetime(ts_num, unit="s", errors="coerce")
        ts_ms = ts_num * 1000
    elif ts_num < 10**14:
        dt = pd.to_datetime(ts_num, unit="ms", errors="coerce")
        ts_ms = ts_num
    else:
        dt = pd.to_datetime(ts_num, unit="us", errors="coerce")
        ts_ms = ts_num // 1000

    if pd.isna(dt):
        return "2026-01-01 00:00:00", 0
    return dt.strftime("%Y-%m-%d %H:%M:%S"), ts_ms


def build_meta_image_map(meta_records: Iterable[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    meta_image_map: Dict[str, Optional[str]] = {}
    for row in tqdm(meta_records, desc="meta", leave=False):
        parent_asin = safe_str(coalesce(row.get("parent_asin"), row.get("asin"))).strip()
        if not parent_asin:
            continue
        image_url = pick_first_url(coalesce(row.get("images"), row.get("image_urls"), row.get("image")))
        if image_url:
            meta_image_map[parent_asin] = image_url
    return meta_image_map


def build_review_text(row: Dict[str, Any]) -> str:
    title = safe_str(coalesce(row.get("title"), row.get("summary"))).strip()
    text = safe_str(coalesce(row.get("text"), row.get("reviewText"), row.get("review_text"))).strip()
    if title and text:
        return f"{title} {text}".strip()
    return (title or text).strip()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Loading local review file...")
    review_records = load_records(args.review_file)
    print(f"Loaded reviews: {len(review_records)}")

    meta_image_map: Dict[str, Optional[str]] = {}
    if args.meta_file:
        print("Loading local meta file...")
        meta_records = load_records(args.meta_file)
        print(f"Loaded metadata rows: {len(meta_records)}")
        print("Building product image map...")
        meta_image_map = build_meta_image_map(meta_records)

    if args.shuffle:
        random.shuffle(review_records)

    selected_records: List[Dict[str, Any]] = []
    user_count: Dict[str, int] = {}
    user_first_ts: Dict[str, int] = {}

    print("Selecting reviews...")
    for row in tqdm(review_records, desc="reviews-pass1"):
        text = build_review_text(row)
        if len(text.strip()) < args.min_text_len:
            continue

        user_id = safe_str(coalesce(row.get("user_id"), row.get("reviewerID"), row.get("reviewer_id")), "unknown_user")
        ts_str, ts_ms = normalize_timestamp(coalesce(row.get("timestamp"), row.get("unixReviewTime"), row.get("reviewTime")))
        row["__normalized_text"] = text
        row["__normalized_ts_str"] = ts_str
        row["__normalized_ts_ms"] = ts_ms
        row["__normalized_user_id"] = user_id

        selected_records.append(row)
        user_count[user_id] = user_count.get(user_id, 0) + 1
        if user_id not in user_first_ts or (ts_ms and ts_ms < user_first_ts[user_id]) or user_first_ts[user_id] == 0:
            user_first_ts[user_id] = ts_ms

        if args.max_reviews and len(selected_records) >= args.max_reviews:
            break

    print(f"Selected reviews: {len(selected_records)}")

    sentiment_rows: List[Dict[str, Any]] = []
    credibility_rows: List[Dict[str, Any]] = []

    print("Creating CSV rows...")
    for idx, row in enumerate(tqdm(selected_records, desc="reviews-pass2")):
        text = row["__normalized_text"]
        rating = safe_float(coalesce(row.get("rating"), row.get("overall")), 3.0)
        timestamp_str = row["__normalized_ts_str"]
        timestamp_ms = row["__normalized_ts_ms"]
        user_id = row["__normalized_user_id"]
        helpful_vote = safe_int(coalesce(row.get("helpful_vote"), row.get("helpful"), row.get("vote")), 0)
        verified_purchase = safe_bool(coalesce(row.get("verified_purchase"), row.get("verified")), False)

        parent_asin = safe_str(coalesce(row.get("parent_asin"), row.get("asin"))).strip()
        review_image_url = pick_first_url(coalesce(row.get("images"), row.get("image"), row.get("image_urls")))
        image_url = review_image_url or meta_image_map.get(parent_asin)

        review_key = f"{user_id}_{parent_asin}_{timestamp_ms}_{idx}"
        review_id = hashlib.md5(review_key.encode("utf-8")).hexdigest()[:16]

        image_path = ""
        if image_url and args.download_images:
            try:
                image_path = download_image(image_url, image_dir, review_id, timeout=args.image_timeout)
            except Exception:
                image_path = ""

        if timestamp_ms > 0 and user_first_ts.get(user_id, 0) > 0:
            first_dt = pd.to_datetime(user_first_ts[user_id], unit="ms", errors="coerce")
            curr_dt = pd.to_datetime(timestamp_ms, unit="ms", errors="coerce")
            if pd.isna(first_dt) or pd.isna(curr_dt):
                user_account_days = 1
            else:
                user_account_days = max((curr_dt - first_dt).days + 1, 1)
        else:
            user_account_days = 1

        senti_label = sentiment_label_from_rating(rating)
        cred_label = credibility_label_heuristic(
            text=text,
            rating=rating,
            helpful_vote=helpful_vote,
            verified_purchase=verified_purchase,
            user_review_count=user_count.get(user_id, 1),
            user_account_days=user_account_days,
        )

        sentiment_rows.append(
            {
                "review_id": review_id,
                "text": text,
                "image_path": image_path,
                "label": senti_label,
            }
        )

        credibility_rows.append(
            {
                "review_id": review_id,
                "text": text,
                "image_path": image_path,
                "rating": rating,
                "timestamp": timestamp_str,
                "user_id": user_id,
                "user_review_count": user_count.get(user_id, 1),
                "user_account_days": user_account_days,
                "helpful_votes": helpful_vote,
                "verified_purchase": int(verified_purchase),
                "label": cred_label,
            }
        )

    sentiment_df = pd.DataFrame(sentiment_rows)
    credibility_df = pd.DataFrame(credibility_rows)

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
