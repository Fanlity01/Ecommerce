from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from src.features.credibility_features import build_credibility_vector, compute_hourly_burst_score
from src.models.credibility_model import CredibilityMLP
from src.models.sentiment_model import MultimodalSentimentModel
from src.utils.io import load_checkpoint


class InferencePipeline:
    def __init__(
        self,
        sentiment_ckpt: str,
        credibility_ckpt: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        sentiment_state = load_checkpoint(sentiment_ckpt, map_location=device)
        model_cfg = sentiment_state["model_config"]

        self.sentiment_model = MultimodalSentimentModel(**model_cfg).to(self.device)
        self.sentiment_model.load_state_dict(sentiment_state["model_state"])
        self.sentiment_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["text_model_name"])
        self.max_length = sentiment_state.get("max_length", 128)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.credibility_model = None
        self.credibility_feature_dim = None
        if credibility_ckpt:
            cred_state = load_checkpoint(credibility_ckpt, map_location=device)
            self.credibility_feature_dim = int(cred_state["feature_dim"])
            self.credibility_model = CredibilityMLP(
                input_dim=self.credibility_feature_dim,
                hidden_dim=cred_state.get("hidden_dim", 64),
                num_labels=2,
                dropout=cred_state.get("dropout", 0.2),
            ).to(self.device)
            self.credibility_model.load_state_dict(cred_state["model_state"])
            self.credibility_model.eval()

    def _prepare_text(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
        }

    def _prepare_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image).unsqueeze(0)
        return image.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        text: str,
        image_path: str,
        rating: float = 5.0,
        timestamp: str = "2026-01-01 00:00:00",
        user_review_count: float = 0.0,
        user_account_days: float = 1.0,
        helpful_votes: float = 0.0,
        verified_purchase: float = 0.0,
    ) -> Dict[str, float]:
        batch = self._prepare_text(text)
        batch["image"] = self._prepare_image(image_path)

        out = self.sentiment_model(batch)
        probs = torch.softmax(out["logits"], dim=-1).squeeze(0).cpu()
        sentiment_id = int(torch.argmax(probs).item())

        result: Dict[str, float | int | str] = {
            "sentiment_label_id": sentiment_id,
            "sentiment_label": ["负面", "中性", "正面"][sentiment_id],
            "sentiment_negative_prob": float(probs[0].item()),
            "sentiment_neutral_prob": float(probs[1].item()),
            "sentiment_positive_prob": float(probs[2].item()),
        }

        if self.credibility_model is not None:
            burst_scores = compute_hourly_burst_score([timestamp])
            feature_vec = build_credibility_vector(
                text=text,
                rating=rating,
                timestamp=timestamp,
                user_review_count=user_review_count,
                user_account_days=user_account_days,
                helpful_votes=helpful_votes,
                verified_purchase=verified_purchase,
                sentiment_probs=probs,
                text_feat=out["text_feat"].squeeze(0).cpu(),
                image_feat=out["image_feat"].squeeze(0).cpu(),
                burst_scores=burst_scores,
            )
            feature_vec = feature_vec.unsqueeze(0).to(self.device)
            cred_logits = self.credibility_model(feature_vec)["logits"]
            cred_probs = torch.softmax(cred_logits, dim=-1).squeeze(0).cpu()
            credibility_id = int(torch.argmax(cred_probs).item())
            result.update(
                {
                    "credibility_label_id": credibility_id,
                    "credibility_label": ["低可信", "高可信"][credibility_id],
                    "credibility_low_prob": float(cred_probs[0].item()),
                    "credibility_high_prob": float(cred_probs[1].item()),
                }
            )

        return result
