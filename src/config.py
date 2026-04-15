from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    text_model_name: str = "bert-base-uncased"
    image_size: int = 224
    hidden_dim: int = 256
    num_sentiment_labels: int = 3
    num_credibility_labels: int = 2
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    max_length: int = 128
    num_workers: int = 0
    seed: int = 42


@dataclass
class PathConfig:
    output_root: Path = Path("outputs")
