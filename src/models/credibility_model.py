from typing import Dict

import torch
import torch.nn as nn


class CredibilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_labels: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(features)
        return {"logits": logits}
