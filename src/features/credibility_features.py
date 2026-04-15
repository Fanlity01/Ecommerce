from __future__ import annotations


from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


def _safe_float(x: float) -> float:
    if x is None:
        return 0.0
    return float(x)


def parse_datetime(value: str) -> datetime:
    value = str(value).strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime(2026, 1, 1, 0, 0, 0)


def count_exclamation(text: str) -> int:
    return text.count("!") + text.count("！")


def lexical_diversity(text: str) -> float:
    chars = [c for c in str(text).strip() if not c.isspace()]
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def text_length(text: str) -> int:
    return len(str(text).strip())


def rating_sentiment_alignment(rating: float, sentiment_probs: torch.Tensor) -> float:
    # 负面/中性/正面
    pred_class = int(torch.argmax(sentiment_probs).item())
    if rating >= 4 and pred_class == 2:
        return 1.0
    if rating <= 2 and pred_class == 0:
        return 1.0
    if 2 < rating < 4 and pred_class == 1:
        return 1.0
    return 0.0


def semantic_similarity(text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    if text_feat.size(-1) != image_feat.size(-1):
        min_dim = min(text_feat.size(-1), image_feat.size(-1))
        text_feat = text_feat[..., :min_dim]
        image_feat = image_feat[..., :min_dim]

    return torch.sum(text_feat * image_feat, dim=-1, keepdim=True)


def compute_hourly_burst_score(all_timestamps: Iterable[str]) -> Dict[str, float]:
    parsed = [parse_datetime(ts) for ts in all_timestamps]
    buckets: Dict[str, int] = {}
    for dt in parsed:
        key = dt.strftime("%Y-%m-%d %H:00:00")
        buckets[key] = buckets.get(key, 0) + 1

    counts = np.array(list(buckets.values()), dtype=np.float32)
    mean = float(counts.mean()) if len(counts) else 0.0
    std = float(counts.std()) if len(counts) else 1.0
    std = std if std > 1e-6 else 1.0

    scores: Dict[str, float] = {}
    for key, cnt in buckets.items():
        z = (cnt - mean) / std
        scores[key] = float(max(z, 0.0))
    return scores


def build_credibility_vector(
    text: str,
    rating: float,
    timestamp: str,
    user_review_count: float,
    user_account_days: float,
    helpful_votes: float,
    verified_purchase: float,
    sentiment_probs: torch.Tensor,
    text_feat: torch.Tensor,
    image_feat: torch.Tensor,
    burst_scores: Dict[str, float],
) -> torch.Tensor:
    dt = parse_datetime(timestamp)
    hour_key = dt.strftime("%Y-%m-%d %H:00:00")
    sim = semantic_similarity(text_feat.unsqueeze(0), image_feat.unsqueeze(0)).squeeze(0)

    numeric = torch.tensor(
        [
            _safe_float(rating) / 5.0,
            float(text_length(text)) / 200.0,
            float(count_exclamation(text)) / 10.0,
            float(lexical_diversity(text)),
            _safe_float(user_review_count) / 100.0,
            _safe_float(user_account_days) / 3650.0,
            _safe_float(helpful_votes) / 100.0,
            _safe_float(verified_purchase),
            float(dt.hour) / 23.0,
            float(burst_scores.get(hour_key, 0.0)),
            float(rating_sentiment_alignment(rating, sentiment_probs)),
        ],
        dtype=torch.float32,
    )
    return torch.cat([numeric, sentiment_probs.detach().cpu(), sim.detach().cpu()], dim=0)
