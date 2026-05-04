from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.core import FeatureSpec, POSITIONS_COMMON_DIR


DEFAULT_BASE_TABLE_PATH = POSITIONS_COMMON_DIR / "base_table.csv"
DEFAULT_MAX_TEAMMATES = 10
DEFAULT_MODEL_DIR = Path("models/positions/set_transformer")
DEFAULT_PREDICTIONS_DIR = Path("output/predictions/positions")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 18
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.10
    teammate_embed_dim: int = 64
    objective_hidden_dim: int = 128
    set_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    num_heads: int = 4
    num_set_blocks: int = 2
    ff_expansion: int = 2
    patience: int = 5
    overfit_patience: int = 0
    max_teammates: int = DEFAULT_MAX_TEAMMATES
    train_size_per_group_split: float = 0.75
    val_size_per_group_split: float = 0.125
    test_size_per_group_split: float = 0.125
    num_workers: int = 0
    label_smoothing: float = 0.0
    objective_num_layers: int = 1
    experiment_group: str = "default"


@dataclass(frozen=True)
class DatasetBundle:
    samples_df: pd.DataFrame
    objective: np.ndarray
    teammates: np.ndarray
    teammate_mask: np.ndarray
    feature_spec: FeatureSpec
    label_names: tuple[str, ...]


@dataclass(frozen=True)
class SplitBundle:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_matches: tuple[str, ...]
    val_matches: tuple[str, ...]
    test_matches: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactPaths:
    output_dir: Path
    checkpoint_path: Path
    history_csv_path: Path
    metrics_json_path: Path
    split_json_path: Path


@dataclass(frozen=True)
class ExpectedRoleSlot:
    input_label: str
    slot_label: str
    allowed_labels: tuple[str, ...]
