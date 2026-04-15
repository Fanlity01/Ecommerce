from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel

from src.models.fusion import GatedFusion


class MultimodalSentimentModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "bert-base-chinese",
        hidden_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_dim = self.text_encoder.config.hidden_size

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.image_dim = 512

        self.fusion = GatedFusion(
            text_dim=self.text_dim,
            image_dim=self.image_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.image_encoder(image)
        return feat.flatten(1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        text_feat = self.encode_text(batch["input_ids"], batch["attention_mask"])
        image_feat = self.encode_image(batch["image"])
        fused = self.fusion(text_feat, image_feat)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "text_feat": text_feat,
            "image_feat": image_feat,
            "fused_feat": fused,
        }
