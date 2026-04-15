import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.features.credibility_features import build_credibility_vector, compute_hourly_burst_score
from src.models.credibility_model import CredibilityMLP
from src.services.pipeline import InferencePipeline
from src.utils.io import ensure_dir, save_checkpoint, save_json
from src.utils.seed import seed_everything


class TensorFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--sentiment_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="outputs/credibility")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


@torch.no_grad()
def build_feature_table(df: pd.DataFrame, pipeline: InferencePipeline, image_root: str | None) -> tuple[torch.Tensor, torch.Tensor]:
    burst_scores = compute_hourly_burst_score(df["timestamp"].astype(str).tolist())
    feature_list = []
    label_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract credibility features"):
        image_path = str(row.get("image_path", ""))
        if image_root and image_path and not Path(image_path).is_absolute():
            image_path = str(Path(image_root) / image_path)

        batch = pipeline._prepare_text(str(row["text"]))
        batch["image"] = pipeline._prepare_image(image_path)
        out = pipeline.sentiment_model(batch)
        probs = torch.softmax(out["logits"], dim=-1).squeeze(0).cpu()

        feature_vec = build_credibility_vector(
            text=str(row["text"]),
            rating=float(row.get("rating", 3.0)),
            timestamp=str(row.get("timestamp", "2026-01-01 00:00:00")),
            user_review_count=float(row.get("user_review_count", 0)),
            user_account_days=float(row.get("user_account_days", 1)),
            helpful_votes=float(row.get("helpful_votes", 0)),
            verified_purchase=float(row.get("verified_purchase", 0)),
            sentiment_probs=probs,
            text_feat=out["text_feat"].squeeze(0).cpu(),
            image_feat=out["image_feat"].squeeze(0).cpu(),
            burst_scores=burst_scores,
        )
        feature_list.append(feature_vec)
        label_list.append(int(row["label"]))

    features = torch.stack(feature_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long)
    return features, labels


def evaluate(model: CredibilityMLP, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)["logits"]
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_count += labels.size(0)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def main() -> None:
    args = parse_args()
    seed_everything(42)

    save_dir = ensure_dir(args.save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = InferencePipeline(sentiment_ckpt=args.sentiment_ckpt, credibility_ckpt=None, device=device)
    df = pd.read_csv(args.train_csv)
    features, labels = build_feature_table(df, pipeline, args.image_root)

    train_idx, val_idx = train_test_split(
        list(range(len(df))),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_dataset = TensorFeatureDataset(features[train_idx], labels[train_idx])
    val_dataset = TensorFeatureDataset(features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = features.size(1)
    model = CredibilityMLP(input_dim=input_dim, hidden_dim=64, num_labels=2, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_correct, running_count = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"credibility epoch {epoch + 1}/{args.epochs}")
        for feat, label in pbar:
            feat, label = feat.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(feat)["logits"]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * label.size(0)
            running_correct += int((logits.argmax(dim=-1) == label).sum().item())
            running_count += label.size(0)
            pbar.set_postfix(
                loss=f"{running_loss / max(running_count, 1):.4f}",
                acc=f"{running_correct / max(running_count, 1):.4f}",
            )

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, torch.device(device))

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(epoch_info)
        print(epoch_info)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                save_dir / "best_credibility.pt",
                {
                    "model_state": model.state_dict(),
                    "feature_dim": input_dim,
                    "hidden_dim": 64,
                    "dropout": 0.2,
                    "best_val_acc": best_acc,
                },
            )

    save_json(save_dir / "history.json", {"history": history, "best_val_acc": best_acc})
    print(f"best credibility model saved to: {save_dir / 'best_credibility.pt'}")


if __name__ == "__main__":
    main()
