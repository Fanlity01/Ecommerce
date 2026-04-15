import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        text_h = self.text_proj(text_feat)
        image_h = self.image_proj(image_feat)
        gate = self.gate(torch.cat([text_h, image_h], dim=-1))
        fused = gate * text_h + (1.0 - gate) * image_h
        return self.out(torch.cat([fused, text_h * image_h], dim=-1))
