import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import ModelConfig, TrainConfig
from src.data.multimodal_dataset import MultimodalSentimentDataset
from src.models.sentiment_model import MultimodalSentimentModel
from src.utils.io import ensure_dir, save_checkpoint, save_json
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="outputs/sentiment")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def evaluate(model: MultimodalSentimentModel, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
            }
            logits = model(inputs)["logits"]
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * labels.size(0)
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_count += labels.size(0)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def main() -> None:
    args = parse_args()
    seed_everything(42)

    model_cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = ensure_dir(args.save_dir)

    full_dataset = MultimodalSentimentDataset(
        csv_path=args.train_csv,
        tokenizer_name=model_cfg.text_model_name,
        image_root=args.image_root,
        max_length=args.max_length,
        training=True,
    )
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=full_dataset.df["label"])

    train_dataset = Subset(full_dataset, train_idx)
    val_base = MultimodalSentimentDataset(
        csv_path=args.train_csv,
        tokenizer_name=model_cfg.text_model_name,
        image_root=args.image_root,
        max_length=args.max_length,
        training=False,
    )
    val_dataset = Subset(val_base, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultimodalSentimentModel(
        text_model_name=model_cfg.text_model_name,
        hidden_dim=model_cfg.hidden_dim,
        num_labels=model_cfg.num_sentiment_labels,
        dropout=model_cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss, running_correct, running_count = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"sentiment epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            labels = batch["label"].to(device)
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
            }

            optimizer.zero_grad()
            logits = model(inputs)["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            running_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            running_count += labels.size(0)
            pbar.set_postfix(
                loss=f"{running_loss / max(running_count, 1):.4f}",
                acc=f"{running_correct / max(running_count, 1):.4f}",
            )

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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
                save_dir / "best_sentiment.pt",
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "text_model_name": model_cfg.text_model_name,
                        "hidden_dim": model_cfg.hidden_dim,
                        "num_labels": model_cfg.num_sentiment_labels,
                        "dropout": model_cfg.dropout,
                    },
                    "max_length": args.max_length,
                    "best_val_acc": best_acc,
                },
            )

    save_json(save_dir / "history.json", {"history": history, "best_val_acc": best_acc})
    print(f"best sentiment model saved to: {save_dir / 'best_sentiment.pt'}")


if __name__ == "__main__":
    main()
