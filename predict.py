import argparse
import json

from src.services.pipeline import InferencePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--rating", type=float, default=5.0)
    parser.add_argument("--timestamp", type=str, default="2026-01-01 00:00:00")
    parser.add_argument("--user_review_count", type=float, default=0.0)
    parser.add_argument("--user_account_days", type=float, default=1.0)
    parser.add_argument("--helpful_votes", type=float, default=0.0)
    parser.add_argument("--verified_purchase", type=float, default=0.0)
    parser.add_argument("--sentiment_ckpt", type=str, required=True)
    parser.add_argument("--credibility_ckpt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = InferencePipeline(
        sentiment_ckpt=args.sentiment_ckpt,
        credibility_ckpt=args.credibility_ckpt,
    )
    result = pipeline.predict(
        text=args.text,
        image_path=args.image,
        rating=args.rating,
        timestamp=args.timestamp,
        user_review_count=args.user_review_count,
        user_account_days=args.user_account_days,
        helpful_votes=args.helpful_votes,
        verified_purchase=args.verified_purchase,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
