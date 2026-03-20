from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from football_ai.core.config import Config
from football_ai.visualization import Drawer

try:
    from .position_dataset import (
        POSITIONS_COMMON_DIR,
        ROLE_LABELS_V1,
        FeatureSpec,
        add_velocity_features,
        build_observations_from_tracks,
        build_role_samples,
        find_project_root,
        infer_attack_direction_by_team,
        load_tracks_json,
        resolve_tracks_path_for_video,
    )
except ImportError:  # pragma: no cover - soporte ejecución directa del archivo.
    import sys

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.positions.position_dataset import (  # type: ignore
        POSITIONS_COMMON_DIR,
        ROLE_LABELS_V1,
        FeatureSpec,
        add_velocity_features,
        build_observations_from_tracks,
        build_role_samples,
        find_project_root,
        infer_attack_direction_by_team,
        load_tracks_json,
        resolve_tracks_path_for_video,
    )


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
    patience: int = 5
    max_teammates: int = DEFAULT_MAX_TEAMMATES
    train_size_per_group_split: float = 0.75
    val_size_per_group_split: float = 0.125
    test_size_per_group_split: float = 0.125
    num_workers: int = 0


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _ordered_label_names(labels: Iterable[str]) -> tuple[str, ...]:
    labels_set = {str(label) for label in labels}
    ordered = [label for label in ROLE_LABELS_V1 if label in labels_set]
    extras = sorted(labels_set.difference(ordered))
    return tuple(ordered + extras)


def load_labeled_dataset_from_base_table(
    project_root: Path | None = None,
    base_table_path: Path | None = None,
    max_teammates: int = DEFAULT_MAX_TEAMMATES,
    prefer_cached_common_dataset: bool = True,
) -> DatasetBundle:
    project_root = find_project_root(project_root)
    path = (project_root / DEFAULT_BASE_TABLE_PATH) if base_table_path is None else Path(base_table_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el base_table etiquetado: {path}")

    if prefer_cached_common_dataset:
        cached_bundle = _load_cached_common_dataset_if_available(path=path, max_teammates=max_teammates)
        if cached_bundle is not None:
            return cached_bundle

    base_df = pd.read_csv(path)
    if "role_label" not in base_df.columns:
        raise ValueError(f"{path} no contiene columna 'role_label'.")

    base_df["role_label"] = base_df["role_label"].astype("string").str.strip()
    base_df.loc[base_df["role_label"] == "", "role_label"] = pd.NA
    base_df = base_df[base_df["role_label"].notna()].copy()
    if base_df.empty:
        raise ValueError(f"{path} no contiene filas etiquetadas.")

    if "team_id" not in base_df.columns or "match_id" not in base_df.columns:
        raise ValueError(f"{path} debe contener 'match_id' y 'team_id'.")

    if "vx" not in base_df.columns or "vy" not in base_df.columns:
        base_df = add_velocity_features(base_df)

    samples_parts: list[pd.DataFrame] = []
    teammate_tensors: list[np.ndarray] = []
    teammate_masks: list[np.ndarray] = []
    feature_spec: FeatureSpec | None = None

    for match_id in sorted(base_df["match_id"].dropna().astype(str).unique().tolist()):
        match_df = base_df[base_df["match_id"].astype(str) == match_id].copy()
        attack_direction_by_team, _ = infer_attack_direction_by_team(match_df)
        samples_df, teammates_tensor, teammate_mask, current_spec = build_role_samples(
            observations_with_roles=match_df,
            attack_direction_by_team=attack_direction_by_team,
            max_teammates=int(max_teammates),
            drop_unlabeled=True,
        )
        if samples_df.empty:
            continue
        if feature_spec is None:
            feature_spec = current_spec
        elif feature_spec != current_spec:
            raise ValueError("FeatureSpec inconsistente entre partidos al reconstruir el dataset.")
        samples_parts.append(samples_df)
        teammate_tensors.append(np.asarray(teammates_tensor, dtype=np.float32))
        teammate_masks.append(np.asarray(teammate_mask, dtype=bool))

    if not samples_parts or feature_spec is None:
        raise ValueError(f"No se pudieron reconstruir samples desde {path}.")

    samples_df = pd.concat(samples_parts, ignore_index=True)
    teammates = np.concatenate(teammate_tensors, axis=0)
    teammate_mask = np.concatenate(teammate_masks, axis=0).astype(bool)
    objective = _safe_float_array(
        samples_df.loc[:, list(feature_spec.objective_feature_names)].to_numpy(dtype=np.float32)
    )
    label_names = _ordered_label_names(samples_df["label"].astype(str).tolist())

    return DatasetBundle(
        samples_df=samples_df.reset_index(drop=True),
        objective=objective,
        teammates=teammates,
        teammate_mask=teammate_mask,
        feature_spec=feature_spec,
        label_names=label_names,
    )


def _load_cached_common_dataset_if_available(
    path: Path,
    max_teammates: int,
) -> DatasetBundle | None:
    samples_csv_path = path.with_name("samples_metadata_and_obj_features.csv")
    samples_npz_path = path.with_name("samples_teammates.npz")
    meta_json_path = path.with_name("dataset_meta.json")
    if not samples_csv_path.exists() or not samples_npz_path.exists() or not meta_json_path.exists():
        return None

    with meta_json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    objective_feature_names = tuple(meta.get("objective_feature_names", []))
    teammate_feature_names = tuple(meta.get("teammate_feature_names", []))
    if not objective_feature_names or not teammate_feature_names:
        return None
    if int(meta.get("max_teammates", max_teammates)) != int(max_teammates):
        return None

    samples_df = pd.read_csv(samples_csv_path)
    npz = np.load(samples_npz_path)
    teammates = np.asarray(npz["teammates_tensor"], dtype=np.float32)
    teammate_mask = np.asarray(npz["teammate_mask"], dtype=bool)
    if len(samples_df) != len(teammates) or len(samples_df) != len(teammate_mask):
        raise ValueError(
            "El cache común de samples está desalineado con el CSV de metadata."
        )

    objective = _safe_float_array(
        samples_df.loc[:, list(objective_feature_names)].to_numpy(dtype=np.float32)
    )
    label_names = _ordered_label_names(samples_df["label"].astype(str).tolist())
    return DatasetBundle(
        samples_df=samples_df.reset_index(drop=True),
        objective=objective,
        teammates=teammates,
        teammate_mask=teammate_mask,
        feature_spec=FeatureSpec(
            objective_feature_names=objective_feature_names,
            teammate_feature_names=teammate_feature_names,
        ),
        label_names=label_names,
    )


def split_dataset_by_match(
    dataset: DatasetBundle,
    seed: int = 42,
) -> SplitBundle:
    groups = dataset.samples_df["match_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError(
            "Se necesitan al menos 3 partidos/clips distintos para train/val/test por grupos."
        )

    all_labels = set(dataset.samples_df["label"].astype(str).tolist())
    indices = np.arange(len(dataset.samples_df))

    for attempt in range(24):
        current_seed = int(seed + attempt)

        first_splitter = GroupShuffleSplit(n_splits=1, test_size=1, random_state=current_seed)
        train_val_idx, test_idx = next(first_splitter.split(indices, groups=groups))
        train_val_groups = groups[train_val_idx]

        second_splitter = GroupShuffleSplit(n_splits=1, test_size=1, random_state=current_seed + 97)
        train_rel_idx, val_rel_idx = next(
            second_splitter.split(train_val_idx, groups=train_val_groups)
        )
        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        train_labels = set(dataset.samples_df.iloc[train_idx]["label"].astype(str).tolist())
        if train_labels != all_labels:
            continue

        return SplitBundle(
            train_idx=np.asarray(train_idx, dtype=np.int64),
            val_idx=np.asarray(val_idx, dtype=np.int64),
            test_idx=np.asarray(test_idx, dtype=np.int64),
            train_matches=tuple(sorted(pd.unique(groups[train_idx]).tolist())),
            val_matches=tuple(sorted(pd.unique(groups[val_idx]).tolist())),
            test_matches=tuple(sorted(pd.unique(groups[test_idx]).tolist())),
        )

    raise RuntimeError("No se pudo construir un split por partido que conserve todas las clases en train.")


class FeatureStandardizer:
    def __init__(
        self,
        objective_mean: np.ndarray,
        objective_std: np.ndarray,
        teammate_mean: np.ndarray,
        teammate_std: np.ndarray,
    ) -> None:
        self.objective_mean = _safe_float_array(objective_mean).reshape(1, -1)
        self.objective_std = _safe_float_array(objective_std).reshape(1, -1)
        self.teammate_mean = _safe_float_array(teammate_mean).reshape(1, 1, -1)
        self.teammate_std = _safe_float_array(teammate_std).reshape(1, 1, -1)

    @classmethod
    def fit(
        cls,
        objective: np.ndarray,
        teammates: np.ndarray,
        teammate_mask: np.ndarray,
    ) -> "FeatureStandardizer":
        objective = _safe_float_array(objective)
        teammates = _safe_float_array(teammates)
        teammate_mask = np.asarray(teammate_mask, dtype=bool)

        objective_mean = objective.mean(axis=0)
        objective_std = objective.std(axis=0)
        objective_std = np.where(objective_std < 1e-6, 1.0, objective_std)

        if teammate_mask.any():
            teammate_values = teammates[teammate_mask]
            teammate_mean = teammate_values.mean(axis=0)
            teammate_std = teammate_values.std(axis=0)
            teammate_std = np.where(teammate_std < 1e-6, 1.0, teammate_std)
        else:
            teammate_dim = teammates.shape[-1]
            teammate_mean = np.zeros((teammate_dim,), dtype=np.float32)
            teammate_std = np.ones((teammate_dim,), dtype=np.float32)

        return cls(
            objective_mean=objective_mean,
            objective_std=objective_std,
            teammate_mean=teammate_mean,
            teammate_std=teammate_std,
        )

    def transform(
        self,
        objective: np.ndarray,
        teammates: np.ndarray,
        teammate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        objective = _safe_float_array(objective)
        teammates = _safe_float_array(teammates)
        teammate_mask = np.asarray(teammate_mask, dtype=bool)

        objective_scaled = (objective - self.objective_mean) / self.objective_std
        teammates_scaled = (teammates - self.teammate_mean) / self.teammate_std
        teammates_scaled = np.where(teammate_mask[..., None], teammates_scaled, 0.0)
        return objective_scaled.astype(np.float32), teammates_scaled.astype(np.float32)

    def to_state(self) -> dict[str, Any]:
        return {
            "objective_mean": self.objective_mean.astype(np.float32).reshape(-1).tolist(),
            "objective_std": self.objective_std.astype(np.float32).reshape(-1).tolist(),
            "teammate_mean": self.teammate_mean.astype(np.float32).reshape(-1).tolist(),
            "teammate_std": self.teammate_std.astype(np.float32).reshape(-1).tolist(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FeatureStandardizer":
        return cls(
            objective_mean=np.asarray(state["objective_mean"], dtype=np.float32),
            objective_std=np.asarray(state["objective_std"], dtype=np.float32),
            teammate_mean=np.asarray(state["teammate_mean"], dtype=np.float32),
            teammate_std=np.asarray(state["teammate_std"], dtype=np.float32),
        )


class RoleDataset(Dataset):
    def __init__(
        self,
        objective: np.ndarray,
        teammates: np.ndarray,
        teammate_mask: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.objective = torch.as_tensor(objective, dtype=torch.float32)
        self.teammates = torch.as_tensor(teammates, dtype=torch.float32)
        self.teammate_mask = torch.as_tensor(teammate_mask, dtype=torch.bool)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.objective.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch = {
            "objective": self.objective[idx],
            "teammates": self.teammates[idx],
            "teammate_mask": self.teammate_mask[idx],
        }
        if self.labels is not None:
            batch["labels"] = self.labels[idx]
        return batch


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embed_dim) * 2, int(embed_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PoolingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, int(embed_dim)))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(embed_dim) * 2, int(embed_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        seeds = self.seed.expand(x.shape[0], -1, -1)
        attn_out, _ = self.attn(
            seeds,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm1(seeds + attn_out)
        pooled = self.norm2(pooled + self.ff(pooled))
        return pooled


class RoleSetTransformer(nn.Module):
    def __init__(
        self,
        objective_dim: int,
        teammate_dim: int,
        num_classes: int,
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.objective_encoder = MLP(
            input_dim=int(objective_dim),
            hidden_dims=(int(config.objective_hidden_dim),),
            output_dim=int(config.set_hidden_dim),
            dropout=float(config.dropout),
        )
        self.teammate_encoder = MLP(
            input_dim=int(teammate_dim),
            hidden_dims=(int(config.teammate_embed_dim),),
            output_dim=int(config.set_hidden_dim),
            dropout=float(config.dropout),
        )
        self.set_blocks = nn.ModuleList(
            [
                SetAttentionBlock(
                    embed_dim=int(config.set_hidden_dim),
                    num_heads=int(config.num_heads),
                    dropout=float(config.dropout),
                )
                for _ in range(int(config.num_set_blocks))
            ]
        )
        self.pool = PoolingMultiheadAttention(
            embed_dim=int(config.set_hidden_dim),
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(config.set_hidden_dim) * 2, int(config.fusion_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.fusion_hidden_dim), int(num_classes)),
        )

    def forward(
        self,
        objective: torch.Tensor,
        teammates: torch.Tensor,
        teammate_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_obj = self.objective_encoder(objective)
        z = self.teammate_encoder(teammates)

        safe_mask = teammate_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
            z = z.clone()
            z[empty_rows, 0, :] = 0.0

        key_padding_mask = ~safe_mask
        for block in self.set_blocks:
            z = block(z, key_padding_mask=key_padding_mask)

        h_set = self.pool(z, key_padding_mask=key_padding_mask).squeeze(1)
        logits = self.classifier(torch.cat([h_obj, h_set], dim=-1))
        return logits


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_label_encoder(label_names: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    label_to_idx = {str(label): idx for idx, label in enumerate(label_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    return label_to_idx, idx_to_label


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    labels_idx = list(range(len(label_names)))
    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels_idx,
        target_names=list(label_names),
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels_idx)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: list[float] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in loader:
        objective = batch["objective"].to(device)
        teammates = batch["teammates"].to(device)
        teammate_mask = batch["teammate_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(objective, teammates, teammate_mask)
        loss = criterion(logits, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        preds.append(logits.detach().cpu().argmax(dim=1).numpy())
        targets.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(targets, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _subset_indices(indices: np.ndarray, max_size: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_size is None or max_size <= 0 or len(indices) <= int(max_size):
        return indices
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(indices, size=int(max_size), replace=False))
    return chosen.astype(np.int64)


def _subset_indices_keep_labels(
    indices: np.ndarray,
    max_size: int | None,
    seed: int,
    labels_idx: np.ndarray,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_size is None or max_size <= 0 or len(indices) <= int(max_size):
        return indices

    labels_idx = np.asarray(labels_idx, dtype=np.int64)
    present_labels = np.unique(labels_idx[indices])
    if int(max_size) < len(present_labels):
        raise ValueError(
            f"max_size={max_size} es menor que el número de clases presentes ({len(present_labels)})."
        )

    rng = np.random.default_rng(int(seed))
    mandatory: list[int] = []
    for label in present_labels.tolist():
        label_indices = indices[labels_idx[indices] == int(label)]
        chosen = int(rng.choice(label_indices, size=1, replace=False)[0])
        mandatory.append(chosen)

    remaining_pool = np.array(
        sorted(set(indices.tolist()).difference(mandatory)),
        dtype=np.int64,
    )
    extra_needed = int(max_size) - len(mandatory)
    if extra_needed > 0 and len(remaining_pool) > 0:
        extra = rng.choice(
            remaining_pool,
            size=min(extra_needed, len(remaining_pool)),
            replace=False,
        )
        combined = np.sort(np.concatenate([np.asarray(mandatory, dtype=np.int64), extra]))
        return combined.astype(np.int64)
    return np.sort(np.asarray(mandatory, dtype=np.int64))


def _build_dataloaders(
    objective_scaled: np.ndarray,
    teammates_scaled: np.ndarray,
    teammate_mask: np.ndarray,
    labels_idx: np.ndarray,
    split: SplitBundle,
    config: TrainingConfig,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, np.ndarray]]:
    train_idx = _subset_indices_keep_labels(
        split.train_idx,
        max_train_samples,
        seed=config.seed,
        labels_idx=labels_idx,
    )
    val_idx = _subset_indices(split.val_idx, max_val_samples, seed=config.seed + 1)
    test_idx = _subset_indices(split.test_idx, max_test_samples, seed=config.seed + 2)

    datasets = {
        "train": RoleDataset(
            objective=objective_scaled[train_idx],
            teammates=teammates_scaled[train_idx],
            teammate_mask=teammate_mask[train_idx],
            labels=labels_idx[train_idx],
        ),
        "val": RoleDataset(
            objective=objective_scaled[val_idx],
            teammates=teammates_scaled[val_idx],
            teammate_mask=teammate_mask[val_idx],
            labels=labels_idx[val_idx],
        ),
        "test": RoleDataset(
            objective=objective_scaled[test_idx],
            teammates=teammates_scaled[test_idx],
            teammate_mask=teammate_mask[test_idx],
            labels=labels_idx[test_idx],
        ),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            shuffle=(name == "train"),
            num_workers=int(config.num_workers),
            drop_last=False,
        )
        for name, dataset in datasets.items()
    }
    return loaders["train"], loaders["val"], loaders["test"], {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def _artifact_paths(
    project_root: Path,
    output_dir: Path | None = None,
) -> ArtifactPaths:
    base_dir = (
        project_root / DEFAULT_MODEL_DIR
        if output_dir is None
        else Path(output_dir)
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        output_dir=run_dir,
        checkpoint_path=run_dir / "set_transformer_checkpoint.pt",
        history_csv_path=run_dir / "training_history.csv",
        metrics_json_path=run_dir / "metrics.json",
        split_json_path=run_dir / "split.json",
    )


def train_position_model(
    project_root: Path | None = None,
    base_table_path: Path | None = None,
    output_dir: Path | None = None,
    config: TrainingConfig | None = None,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    prefer_cached_common_dataset: bool = True,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    config = config or TrainingConfig()
    set_global_seed(config.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    dataset = load_labeled_dataset_from_base_table(
        project_root=project_root,
        base_table_path=base_table_path,
        max_teammates=int(config.max_teammates),
        prefer_cached_common_dataset=bool(prefer_cached_common_dataset),
    )
    split = split_dataset_by_match(dataset, seed=config.seed)

    label_to_idx, idx_to_label = _build_label_encoder(dataset.label_names)
    labels_idx = dataset.samples_df["label"].astype(str).map(label_to_idx).to_numpy(dtype=np.int64)

    standardizer = FeatureStandardizer.fit(
        objective=dataset.objective[split.train_idx],
        teammates=dataset.teammates[split.train_idx],
        teammate_mask=dataset.teammate_mask[split.train_idx],
    )
    objective_scaled, teammates_scaled = standardizer.transform(
        objective=dataset.objective,
        teammates=dataset.teammates,
        teammate_mask=dataset.teammate_mask,
    )

    train_loader, val_loader, test_loader, used_indices = _build_dataloaders(
        objective_scaled=objective_scaled,
        teammates_scaled=teammates_scaled,
        teammate_mask=dataset.teammate_mask,
        labels_idx=labels_idx,
        split=split,
        config=config,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )

    device = _select_device()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(dataset.label_names)),
        y=labels_idx[used_indices["train_idx"]],
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = RoleSetTransformer(
        objective_dim=int(dataset.objective.shape[1]),
        teammate_dim=int(dataset.teammates.shape[2]),
        num_classes=len(dataset.label_names),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    history_rows: list[dict[str, Any]] = []
    best_val_macro_f1 = -np.inf
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_epoch = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_epoch = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        train_metrics = _classification_metrics(
            y_true=train_epoch["y_true"],
            y_pred=train_epoch["y_pred"],
            label_names=dataset.label_names,
        )
        val_metrics = _classification_metrics(
            y_true=val_epoch["y_true"],
            y_pred=val_epoch["y_pred"],
            label_names=dataset.label_names,
        )

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_epoch["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_loss": float(val_epoch["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if float(val_metrics["macro_f1"]) > float(best_val_macro_f1):
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(config.patience):
                break

    if best_state is None:
        raise RuntimeError("El entrenamiento no produjo ningún checkpoint válido.")

    model.load_state_dict(best_state)
    train_final = _run_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    val_final = _run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    test_final = _run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    train_metrics = _classification_metrics(
        y_true=train_final["y_true"],
        y_pred=train_final["y_pred"],
        label_names=dataset.label_names,
    )
    val_metrics = _classification_metrics(
        y_true=val_final["y_true"],
        y_pred=val_final["y_pred"],
        label_names=dataset.label_names,
    )
    test_metrics = _classification_metrics(
        y_true=test_final["y_true"],
        y_pred=test_final["y_pred"],
        label_names=dataset.label_names,
    )

    artifacts = _artifact_paths(project_root=project_root, output_dir=output_dir)
    pd.DataFrame(history_rows).to_csv(artifacts.history_csv_path, index=False)

    split_payload = {
        "train_matches": list(split.train_matches),
        "val_matches": list(split.val_matches),
        "test_matches": list(split.test_matches),
        "num_train_samples": int(len(used_indices["train_idx"])),
        "num_val_samples": int(len(used_indices["val_idx"])),
        "num_test_samples": int(len(used_indices["test_idx"])),
    }
    with artifacts.split_json_path.open("w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)

    metrics_payload = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "num_classes_trained": int(len(dataset.label_names)),
        "trained_labels": list(dataset.label_names),
        "missing_known_labels": [
            label for label in ROLE_LABELS_V1 if label not in set(dataset.label_names)
        ],
        "used_cached_common_dataset": bool(prefer_cached_common_dataset),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with artifacts.metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    checkpoint = {
        "created_at": datetime.now().isoformat(),
        "model_state_dict": model.state_dict(),
        "training_config": asdict(config),
        "objective_feature_names": list(dataset.feature_spec.objective_feature_names),
        "teammate_feature_names": list(dataset.feature_spec.teammate_feature_names),
        "label_names": list(dataset.label_names),
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "standardizer": standardizer.to_state(),
        "best_epoch": int(best_epoch),
        "metrics_path": str(artifacts.metrics_json_path),
        "heuristics": {
            "goalkeeper_role": "POR",
        },
        "used_cached_common_dataset": bool(prefer_cached_common_dataset),
    }
    torch.save(checkpoint, artifacts.checkpoint_path)

    return {
        "artifacts": artifacts,
        "metrics": metrics_payload,
        "history_df": pd.DataFrame(history_rows),
        "split": split_payload,
        "checkpoint_path": artifacts.checkpoint_path,
    }


def _load_checkpoint(model_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(model_path), map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint inválido: {model_path}")
    return dict(checkpoint)


def _prepare_inference_dataset(
    project_root: Path,
    video_path: Path,
    tracks_path: Path | None = None,
    max_teammates: int = DEFAULT_MAX_TEAMMATES,
) -> tuple[DatasetBundle, pd.DataFrame, pd.DataFrame, dict[str, int], Path]:
    resolved_tracks_path = tracks_path or resolve_tracks_path_for_video(project_root, video_path)
    if not resolved_tracks_path.exists():
        raise FileNotFoundError(f"No existe tracks JSON para inferencia: {resolved_tracks_path}")

    tracks = load_tracks_json(resolved_tracks_path)
    observations = build_observations_from_tracks(
        tracks=tracks,
        match_id=video_path.stem.replace(" ", "_"),
        tracked_classes=("player", "goalkeeper"),
    )
    observations = add_velocity_features(observations)
    observations["role_label"] = pd.NA
    if observations.empty:
        raise ValueError("No hay observaciones con field_position_m para inferencia.")

    attack_direction_by_team, attack_details = infer_attack_direction_by_team(observations)
    samples_df, teammates_tensor, teammate_mask, feature_spec = build_role_samples(
        observations_with_roles=observations,
        attack_direction_by_team=attack_direction_by_team,
        max_teammates=int(max_teammates),
        drop_unlabeled=False,
    )
    if samples_df.empty:
        raise ValueError("No se pudieron construir samples de inferencia.")

    meta_cols = [
        "match_id",
        "frame_id",
        "team_id",
        "player_id",
        "class_name",
        "x_m",
        "y_m",
        "confidence_tracking",
    ]
    sample_meta = observations[meta_cols].drop_duplicates(
        subset=["match_id", "frame_id", "team_id", "player_id"],
        keep="last",
    )
    samples_df = samples_df.merge(
        sample_meta,
        on=["match_id", "frame_id", "team_id", "player_id"],
        how="left",
    )
    objective = _safe_float_array(
        samples_df.loc[:, list(feature_spec.objective_feature_names)].to_numpy(dtype=np.float32)
    )
    bundle = DatasetBundle(
        samples_df=samples_df.reset_index(drop=True),
        objective=objective,
        teammates=np.asarray(teammates_tensor, dtype=np.float32),
        teammate_mask=np.asarray(teammate_mask, dtype=bool),
        feature_spec=feature_spec,
        label_names=tuple(),
    )
    return bundle, observations, attack_details, attack_direction_by_team, resolved_tracks_path


def _predict_probabilities(
    model: nn.Module,
    objective: np.ndarray,
    teammates: np.ndarray,
    teammate_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = RoleDataset(
        objective=objective,
        teammates=teammates,
        teammate_mask=teammate_mask,
        labels=None,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["objective"].to(device),
                batch["teammates"].to(device),
                batch["teammate_mask"].to(device),
            )
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


ROLE_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "GK": ("POR",),
    "GOALKEEPER": ("POR",),
    "KEEPER": ("POR",),
    "DFIZQ": ("DFC_IZQ",),
    "DF_IZQ": ("DFC_IZQ",),
    "DFCIZQ": ("DFC_IZQ",),
    "DFDCHA": ("DFC_DER",),
    "DF_DCHA": ("DFC_DER",),
    "DFC_DCHA": ("DFC_DER",),
    "DFDER": ("DFC_DER",),
    "DF_DER": ("DFC_DER",),
    "DFCENT": ("DFC_CENT",),
    "DFCENT": ("DFC_CENT",),
    "DFC_CENTRAL": ("DFC_CENT",),
    "DEL": ("DC",),
    "ST": ("DC",),
    "STRIKER": ("DC",),
}


def _normalize_role_token(value: Any) -> str:
    token = str(value).strip().upper()
    token = token.replace("-", "_").replace(" ", "_")
    token = "_".join(part for part in token.split("_") if part)
    return token


def _resolve_expected_role_slot(
    role_label: Any,
    label_names: Sequence[str],
) -> ExpectedRoleSlot:
    slot_label = _normalize_role_token(role_label)
    known_labels = {str(label) for label in label_names}

    if slot_label in known_labels or slot_label == "POR":
        return ExpectedRoleSlot(
            input_label=str(role_label),
            slot_label=slot_label,
            allowed_labels=(slot_label,),
        )

    allowed = ROLE_SLOT_ALIASES.get(slot_label)
    if allowed is None:
        supported = sorted(set(known_labels).union(ROLE_SLOT_ALIASES.keys(), {"POR"}))
        raise ValueError(
            f"Rol esperado no soportado: {role_label!r}. "
            f"Usa labels del modelo o aliases soportados: {supported}"
        )

    filtered = tuple(label for label in allowed if label == "POR" or label in known_labels)
    if not filtered:
        raise ValueError(
            f"El rol esperado {role_label!r} no es compatible con las clases del checkpoint."
        )
    return ExpectedRoleSlot(
        input_label=str(role_label),
        slot_label=slot_label,
        allowed_labels=filtered,
    )


def _aggregate_player_predictions(
    frame_df: pd.DataFrame,
    label_names: Sequence[str],
) -> pd.DataFrame:
    prob_cols = [f"prob_{label}" for label in label_names]
    rows: list[dict[str, Any]] = []
    grouped = frame_df.groupby(["team_id", "player_id"], sort=True)
    for (team_id, player_id), group in grouped:
        class_name_mode = group["class_name"].dropna().astype(str).mode()
        class_name = class_name_mode.iloc[0] if not class_name_mode.empty else "player"

        if class_name == "goalkeeper":
            row = {
                "team_id": str(team_id),
                "player_id": int(player_id),
                "class_name": class_name,
                "predicted_role": "POR",
                "predicted_role_confidence": 1.0,
                "frames_seen": int(len(group)),
            }
            for prob_col in prob_cols:
                row[prob_col] = 0.0
            rows.append(row)
            continue

        mean_probs = group[prob_cols].mean(axis=0)
        pred_col = str(mean_probs.idxmax())
        pred_label = pred_col.replace("prob_", "", 1)
        row = {
            "team_id": str(team_id),
            "player_id": int(player_id),
            "class_name": class_name,
            "predicted_role": pred_label,
            "predicted_role_confidence": float(mean_probs.max()),
            "frames_seen": int(len(group)),
        }
        row.update({prob_col: float(mean_probs[prob_col]) for prob_col in prob_cols})
        rows.append(row)

    return pd.DataFrame(rows)


def _apply_expected_roles_constraint(
    player_predictions_df: pd.DataFrame,
    label_names: Sequence[str],
    expected_roles_by_team: Mapping[str, Sequence[str]] | None,
) -> pd.DataFrame:
    constrained = player_predictions_df.copy()
    constrained["predicted_role_unconstrained"] = constrained["predicted_role"].astype(str)
    constrained["predicted_role_confidence_unconstrained"] = pd.to_numeric(
        constrained["predicted_role_confidence"],
        errors="coerce",
    ).astype(np.float32)
    constrained["expected_role_slot"] = pd.Series(
        [pd.NA] * len(constrained),
        dtype="string",
    )
    constrained["assignment_method"] = pd.Series(
        ["unconstrained"] * len(constrained),
        dtype="string",
    )
    constrained["assignment_cost"] = np.nan

    if not expected_roles_by_team:
        return constrained

    available_team_ids = set(constrained["team_id"].astype(str).unique().tolist())
    unknown_team_ids = sorted(
        {str(team_id) for team_id in expected_roles_by_team.keys()}.difference(available_team_ids)
    )
    if unknown_team_ids:
        raise ValueError(
            f"expected_roles_by_team contiene team_id inexistentes en la inferencia: {unknown_team_ids}"
        )

    epsilon = 1e-9
    prob_cols = [f"prob_{label}" for label in label_names]
    for raw_team_id, expected_roles in expected_roles_by_team.items():
        team_id = str(raw_team_id)
        team_mask = constrained["team_id"].astype(str) == team_id
        team_df = constrained.loc[team_mask].copy()
        slots = [
            _resolve_expected_role_slot(role_label=role_label, label_names=label_names)
            for role_label in expected_roles
        ]

        goalkeeper_slots = [slot for slot in slots if "POR" in slot.allowed_labels]
        field_slots = [slot for slot in slots if "POR" not in slot.allowed_labels]

        goalkeeper_df = team_df[team_df["class_name"].astype(str) == "goalkeeper"].copy()
        field_df = team_df[team_df["class_name"].astype(str) != "goalkeeper"].copy()

        if len(goalkeeper_df) > len(goalkeeper_slots):
            raise ValueError(
                f"El equipo {team_id!r} tiene {len(goalkeeper_df)} goalkeeper tracks y "
                f"{len(goalkeeper_slots)} slots POR esperados."
            )
        if goalkeeper_slots:
            goalkeeper_df = goalkeeper_df.sort_values(
                ["frames_seen", "player_id"],
                ascending=[False, True],
            )
            for (_, row), slot in zip(goalkeeper_df.iterrows(), goalkeeper_slots):
                constrained.at[row.name, "predicted_role"] = "POR"
                constrained.at[row.name, "predicted_role_confidence"] = 1.0
                constrained.at[row.name, "expected_role_slot"] = slot.slot_label
                constrained.at[row.name, "assignment_method"] = "hungarian_expected_roles"
                constrained.at[row.name, "assignment_cost"] = 0.0

        if not field_slots:
            continue

        field_indices = field_df.index.tolist()
        cost_matrix = np.zeros((len(field_indices), len(field_slots)), dtype=np.float64)
        best_labels_per_pair: list[list[tuple[str, float]]] = []

        for row_pos, row_idx in enumerate(field_indices):
            player_row = field_df.loc[row_idx]
            row_pairs: list[tuple[str, float]] = []
            for col_pos, slot in enumerate(field_slots):
                slot_probs = []
                for label in slot.allowed_labels:
                    if label == "POR":
                        continue
                    prob_value = float(player_row.get(f"prob_{label}", 0.0))
                    slot_probs.append((label, prob_value))
                if not slot_probs:
                    raise ValueError(
                        f"El slot {slot.slot_label!r} del equipo {team_id!r} no tiene clases válidas."
                    )
                best_label, best_prob = max(slot_probs, key=lambda item: item[1])
                row_pairs.append((best_label, best_prob))
                cost_matrix[row_pos, col_pos] = -math.log(max(best_prob, epsilon))
            best_labels_per_pair.append(row_pairs)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_row_positions = set(row_ind.tolist())
        for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist()):
            row_idx = field_indices[row_pos]
            slot = field_slots[col_pos]
            best_label, best_prob = best_labels_per_pair[row_pos][col_pos]
            constrained.at[row_idx, "predicted_role"] = best_label
            constrained.at[row_idx, "predicted_role_confidence"] = float(best_prob)
            constrained.at[row_idx, "expected_role_slot"] = slot.slot_label
            constrained.at[row_idx, "assignment_method"] = "hungarian_expected_roles"
            constrained.at[row_idx, "assignment_cost"] = float(cost_matrix[row_pos, col_pos])

        fallback_options: list[tuple[str, str]] = []
        seen_fallback_pairs: set[tuple[str, str]] = set()
        for slot in field_slots:
            for label in slot.allowed_labels:
                if label == "POR":
                    continue
                pair = (slot.slot_label, str(label))
                if pair in seen_fallback_pairs:
                    continue
                fallback_options.append(pair)
                seen_fallback_pairs.add(pair)

        for row_pos, row_idx in enumerate(field_indices):
            if row_pos in assigned_row_positions:
                continue
            player_row = field_df.loc[row_idx]
            if not fallback_options:
                constrained.at[row_idx, "assignment_method"] = "expected_roles_unassigned"
                continue

            best_slot_label = None
            best_label = None
            best_prob = -1.0
            for slot_label, label in fallback_options:
                prob_value = float(player_row.get(f"prob_{label}", 0.0))
                if prob_value > best_prob:
                    best_slot_label = slot_label
                    best_label = label
                    best_prob = prob_value

            if best_label is None or best_slot_label is None:
                constrained.at[row_idx, "assignment_method"] = "expected_roles_unassigned"
                continue

            constrained.at[row_idx, "predicted_role"] = best_label
            constrained.at[row_idx, "predicted_role_confidence"] = float(best_prob)
            constrained.at[row_idx, "expected_role_slot"] = best_slot_label
            constrained.at[row_idx, "assignment_method"] = "expected_roles_fallback_best_allowed"
            constrained.at[row_idx, "assignment_cost"] = float(-math.log(max(best_prob, epsilon)))

    keep_cols = [
        "team_id",
        "player_id",
        "class_name",
        "predicted_role",
        "predicted_role_confidence",
        "predicted_role_unconstrained",
        "predicted_role_confidence_unconstrained",
        "expected_role_slot",
        "assignment_method",
        "assignment_cost",
        "frames_seen",
        *prob_cols,
    ]
    available_cols = [col for col in keep_cols if col in constrained.columns]
    return constrained.loc[:, available_cols]


def _augment_tracks_with_predictions(
    tracks: Mapping[str, Any],
    frame_predictions_df: pd.DataFrame,
    player_predictions_df: pd.DataFrame,
) -> dict[str, Any]:
    output = copy.deepcopy(tracks)
    frame_map = {
        (int(row.frame_id), str(row.team_id), int(row.player_id)): (
            str(row.predicted_role_frame),
            float(row.predicted_role_frame_confidence),
        )
        for row in frame_predictions_df.itertuples(index=False)
    }
    player_map = {}
    for row in player_predictions_df.itertuples(index=False):
        stable_payload = {
            "predicted_role": str(row.predicted_role),
            "predicted_role_confidence": float(row.predicted_role_confidence),
        }
        if hasattr(row, "predicted_role_unconstrained"):
            stable_payload["predicted_role_unconstrained"] = str(row.predicted_role_unconstrained)
        if hasattr(row, "predicted_role_confidence_unconstrained"):
            stable_payload["predicted_role_confidence_unconstrained"] = float(
                row.predicted_role_confidence_unconstrained
            )
        if hasattr(row, "expected_role_slot") and pd.notna(row.expected_role_slot):
            stable_payload["expected_role_slot"] = str(row.expected_role_slot)
        if hasattr(row, "assignment_method") and pd.notna(row.assignment_method):
            stable_payload["assignment_method"] = str(row.assignment_method)
        player_map[(str(row.team_id), int(row.player_id))] = stable_payload

    for class_name in ("player", "goalkeeper"):
        frames = output.get(class_name, [])
        if not isinstance(frames, list):
            continue
        for frame_id, frame_tracks in enumerate(frames):
            if not isinstance(frame_tracks, Mapping):
                continue
            for player_id_raw, track_data in list(frame_tracks.items()):
                try:
                    player_id = int(player_id_raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(track_data, Mapping):
                    continue
                team_id_raw = track_data.get("team")
                if team_id_raw is None:
                    continue
                team_id = str(team_id_raw)

                enriched = dict(track_data)
                stable = player_map.get((team_id, player_id))
                frame_pred = frame_map.get((int(frame_id), team_id, player_id))
                if stable is not None:
                    enriched.update(stable)
                if frame_pred is not None:
                    enriched["predicted_role_frame"] = frame_pred[0]
                    enriched["predicted_role_frame_confidence"] = frame_pred[1]
                frame_tracks[player_id_raw] = enriched
    return output


def predict_roles_for_video(
    model_path: Path,
    video_path: Path,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    tracks_path: Path | None = None,
    batch_size: int = 1024,
    expected_roles_by_team: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    video_path = Path(video_path)
    device = _select_device()
    checkpoint = _load_checkpoint(model_path=model_path, device=device)

    config = TrainingConfig(**checkpoint["training_config"])
    label_names = tuple(str(label) for label in checkpoint["label_names"])
    standardizer = FeatureStandardizer.from_state(checkpoint["standardizer"])

    dataset, observations, attack_details, attack_direction_by_team, resolved_tracks_path = (
        _prepare_inference_dataset(
            project_root=project_root,
            video_path=video_path,
            tracks_path=tracks_path,
            max_teammates=int(config.max_teammates),
        )
    )

    if list(dataset.feature_spec.objective_feature_names) != list(checkpoint["objective_feature_names"]):
        raise ValueError("Las objective_feature_names de inferencia no coinciden con el checkpoint.")
    if list(dataset.feature_spec.teammate_feature_names) != list(checkpoint["teammate_feature_names"]):
        raise ValueError("Las teammate_feature_names de inferencia no coinciden con el checkpoint.")

    objective_scaled, teammates_scaled = standardizer.transform(
        objective=dataset.objective,
        teammates=dataset.teammates,
        teammate_mask=dataset.teammate_mask,
    )

    model = RoleSetTransformer(
        objective_dim=int(objective_scaled.shape[1]),
        teammate_dim=int(teammates_scaled.shape[2]),
        num_classes=int(len(label_names)),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    probs = _predict_probabilities(
        model=model,
        objective=objective_scaled,
        teammates=teammates_scaled,
        teammate_mask=dataset.teammate_mask,
        batch_size=int(batch_size),
        device=device,
    )
    pred_idx = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)

    frame_predictions_df = dataset.samples_df.copy()
    for idx, label in enumerate(label_names):
        frame_predictions_df[f"prob_{label}"] = probs[:, idx]
    frame_predictions_df["predicted_role_frame"] = [label_names[int(idx)] for idx in pred_idx]
    frame_predictions_df["predicted_role_frame_confidence"] = pred_conf.astype(np.float32)
    frame_predictions_df.loc[
        frame_predictions_df["class_name"].astype(str) == "goalkeeper",
        "predicted_role_frame",
    ] = "POR"
    frame_predictions_df.loc[
        frame_predictions_df["class_name"].astype(str) == "goalkeeper",
        "predicted_role_frame_confidence",
    ] = 1.0

    player_predictions_df = _aggregate_player_predictions(
        frame_df=frame_predictions_df,
        label_names=label_names,
    )
    player_predictions_df = _apply_expected_roles_constraint(
        player_predictions_df=player_predictions_df,
        label_names=label_names,
        expected_roles_by_team=expected_roles_by_team,
    )
    frame_predictions_df = frame_predictions_df.merge(
        player_predictions_df[
            [
                "team_id",
                "player_id",
                "predicted_role",
                "predicted_role_confidence",
                "predicted_role_unconstrained",
                "predicted_role_confidence_unconstrained",
                "expected_role_slot",
                "assignment_method",
                "assignment_cost",
            ]
        ],
        on=["team_id", "player_id"],
        how="left",
    )

    with resolved_tracks_path.open("r", encoding="utf-8") as f:
        tracks_payload = json.load(f)
    augmented_tracks = _augment_tracks_with_predictions(
        tracks=tracks_payload,
        frame_predictions_df=frame_predictions_df,
        player_predictions_df=player_predictions_df,
    )

    output_base = (
        project_root / DEFAULT_PREDICTIONS_DIR
        if output_dir is None
        else Path(output_dir)
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_base / f"{video_path.stem}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_csv_path = out_dir / "frame_role_predictions.csv"
    player_csv_path = out_dir / "player_role_summary.csv"
    tracks_json_path = out_dir / "tracks_with_predicted_roles.json"
    meta_json_path = out_dir / "inference_meta.json"

    frame_predictions_df.to_csv(frame_csv_path, index=False)
    player_predictions_df.to_csv(player_csv_path, index=False)
    with tracks_json_path.open("w", encoding="utf-8") as f:
        json.dump(augmented_tracks, f, ensure_ascii=False, indent=2)

    meta_payload = {
        "created_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "video_path": str(video_path),
        "tracks_path": str(resolved_tracks_path),
        "output_dir": str(out_dir),
        "device": str(device),
        "num_frame_predictions": int(len(frame_predictions_df)),
        "num_player_predictions": int(len(player_predictions_df)),
        "trained_labels": list(label_names),
        "goalkeeper_heuristic_label": checkpoint.get("heuristics", {}).get("goalkeeper_role", "POR"),
        "expected_roles_by_team": (
            {str(team_id): [str(role) for role in roles] for team_id, roles in expected_roles_by_team.items()}
            if expected_roles_by_team
            else None
        ),
        "stable_assignment_method": (
            "hungarian_expected_roles" if expected_roles_by_team else "unconstrained_softmax"
        ),
        "attack_direction_by_team": {
            str(team): int(direction) for team, direction in attack_direction_by_team.items()
        },
        "attack_direction_details": attack_details.to_dict(orient="records"),
    }
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    return {
        "output_dir": out_dir,
        "frame_predictions_path": frame_csv_path,
        "player_predictions_path": player_csv_path,
        "tracks_with_roles_path": tracks_json_path,
        "meta_path": meta_json_path,
        "frame_predictions_df": frame_predictions_df,
        "player_predictions_df": player_predictions_df,
    }


def _visualization_colors_from_config(project_root: Path) -> dict[str, tuple[int, int, int]]:
    config = Config.from_yaml(project_root / "config.yaml")
    colors_raw = config.get("visualization", "colors", default={}) or {}
    colors: dict[str, tuple[int, int, int]] = {}
    for class_name, color_values in colors_raw.items():
        if not isinstance(color_values, (list, tuple)) or len(color_values) < 3:
            continue
        colors[str(class_name)] = tuple(int(v) for v in color_values[:3])
    if not colors:
        colors = {
            "player": (0, 255, 0),
            "goalkeeper": (0, 255, 255),
            "referee": (255, 0, 0),
            "ball": (0, 0, 255),
        }
    return colors


def render_role_video(
    video_path: Path,
    tracks_path: Path,
    project_root: Path | None = None,
    output_path: Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    video_path = Path(video_path)
    tracks_path = Path(tracks_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el vídeo de entrada: {video_path}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON enriquecido: {tracks_path}")

    with tracks_path.open("r", encoding="utf-8") as f:
        tracks = json.load(f)

    if output_path is None:
        output_path = tracks_path.with_name(f"{video_path.stem}_roles_annotated.mp4")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drawer = Drawer(colors=_visualization_colors_from_config(project_root))
    drawer.draw_tracks(
        tracks=tracks,
        video=str(video_path),
        output_path=str(output_path),
        show=bool(show),
        window_name="Roles Posicionales",
    )
    return {
        "video_path": output_path,
        "tracks_path": tracks_path,
        "source_video_path": video_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrenamiento e inferencia de un Set Transformer para roles posicionales."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrenar modelo desde base_table.csv.")
    train_parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Raíz del proyecto. Por defecto se autodetecta.",
    )
    train_parser.add_argument(
        "--base-table-path",
        type=Path,
        default=None,
        help="Ruta al base_table.csv etiquetado. Por defecto usa el dataset común.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio base donde guardar el checkpoint.",
    )
    train_parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train_parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    train_parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    train_parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    train_parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    train_parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-val-samples", type=int, default=None)
    train_parser.add_argument("--max-test-samples", type=int, default=None)
    train_parser.add_argument(
        "--rebuild-from-base-table",
        action="store_true",
        help="Ignora el cache común de samples y reconstruye el dataset derivado desde base_table.csv.",
    )

    predict_parser = subparsers.add_parser("predict", help="Aplicar checkpoint a un vídeo.")
    predict_parser.add_argument("--project-root", type=Path, default=None)
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--video-path", type=Path, required=True)
    predict_parser.add_argument("--tracks-path", type=Path, default=None)
    predict_parser.add_argument("--output-dir", type=Path, default=None)
    predict_parser.add_argument("--batch-size", type=int, default=1024)

    render_parser = subparsers.add_parser(
        "render-video",
        help="Renderizar un MP4 anotado usando tracks JSON con roles predichos.",
    )
    render_parser.add_argument("--project-root", type=Path, default=None)
    render_parser.add_argument("--video-path", type=Path, required=True)
    render_parser.add_argument("--tracks-path", type=Path, required=True)
    render_parser.add_argument("--output-path", type=Path, default=None)
    render_parser.add_argument("--show", action="store_true")

    return parser


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = _training_config_from_args(args)
        result = train_position_model(
            project_root=args.project_root,
            base_table_path=args.base_table_path,
            output_dir=args.output_dir,
            config=config,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            prefer_cached_common_dataset=not bool(args.rebuild_from_base_table),
        )
        summary = {
            "checkpoint_path": str(result["checkpoint_path"]),
            "output_dir": str(result["artifacts"].output_dir),
            "best_epoch": int(result["metrics"]["best_epoch"]),
            "val_macro_f1": float(result["metrics"]["val"]["macro_f1"]),
            "test_macro_f1": float(result["metrics"]["test"]["macro_f1"]),
            "trained_labels": result["metrics"]["trained_labels"],
            "missing_known_labels": result["metrics"]["missing_known_labels"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "predict":
        result = predict_roles_for_video(
            model_path=args.model_path,
            video_path=args.video_path,
            project_root=args.project_root,
            output_dir=args.output_dir,
            tracks_path=args.tracks_path,
            batch_size=int(args.batch_size),
        )
        summary = {
            "output_dir": str(result["output_dir"]),
            "frame_predictions_path": str(result["frame_predictions_path"]),
            "player_predictions_path": str(result["player_predictions_path"]),
            "tracks_with_roles_path": str(result["tracks_with_roles_path"]),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "render-video":
        result = render_role_video(
            video_path=args.video_path,
            tracks_path=args.tracks_path,
            project_root=args.project_root,
            output_path=args.output_path,
            show=bool(args.show),
        )
        summary = {
            "video_path": str(result["video_path"]),
            "tracks_path": str(result["tracks_path"]),
            "source_video_path": str(result["source_video_path"]),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.error("Comando no soportado.")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
