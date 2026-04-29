from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from football_ai.positions.position_dataset import ROLE_LABELS_V1
from football_ai.positions.set_transformer_pipeline import (
    DEFAULT_BASE_TABLE_PATH,
    DEFAULT_MAX_TEAMMATES,
    DatasetBundle,
    FeatureStandardizer,
    RoleDataset,
    _build_label_encoder,
    _classification_metrics,
    _run_epoch,
    _select_device,
    find_project_root,
    load_labeled_dataset_from_base_table,
    set_global_seed,
    split_dataset_by_match,
)


DEFAULT_OUTPUT_DIR = Path("models/positions/grid_search_outputs")
DEFAULT_FINAL_TRAIN_VAL_OUTPUT_DIR = Path("models/positions/final_train_val")
DEFAULT_BEST_HYPERPARAMETERS_PATH = Path("models/positions/20260427_002133/best_hyperparameters.json")
WINGBACK_TO_FULLBACK_LABEL_MAP = {
    "CI": "LI",
    "CD": "LD",
}


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.10
    label_smoothing: float = 0.0
    teammate_embed_dim: int = 64
    objective_hidden_dim: int = 128
    objective_num_layers: int = 1
    set_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    num_heads: int = 4
    num_set_blocks: int = 2
    ff_expansion: int = 2
    patience: int = 3
    overfit_patience: int = 3
    max_teammates: int = DEFAULT_MAX_TEAMMATES
    train_size_per_group_split: float = 0.75
    val_size_per_group_split: float = 0.125
    test_size_per_group_split: float = 0.125
    num_workers: int = 0
    experiment_group: str = "default"


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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ff_expansion: int) -> None:
        super().__init__()
        ff_dim = int(embed_dim) * int(ff_expansion)
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), ff_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, int(embed_dim)),
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ff_expansion: int) -> None:
        super().__init__()
        ff_dim = int(embed_dim) * int(ff_expansion)
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
            nn.Linear(int(embed_dim), ff_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, int(embed_dim)),
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
        if int(config.set_hidden_dim) % int(config.num_heads) != 0:
            raise ValueError(
                "set_hidden_dim debe ser divisible entre num_heads "
                f"({config.set_hidden_dim=} {config.num_heads=})."
            )
        if int(config.objective_num_layers) < 1:
            raise ValueError("objective_num_layers debe ser >= 1.")
        if int(config.ff_expansion) < 1:
            raise ValueError("ff_expansion debe ser >= 1.")

        objective_hidden_dims = tuple(
            int(config.objective_hidden_dim)
            for _ in range(int(config.objective_num_layers))
        )
        self.objective_encoder = MLP(
            input_dim=int(objective_dim),
            hidden_dims=objective_hidden_dims,
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
                    ff_expansion=int(config.ff_expansion),
                )
                for _ in range(int(config.num_set_blocks))
            ]
        )
        self.pool = PoolingMultiheadAttention(
            embed_dim=int(config.set_hidden_dim),
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            ff_expansion=int(config.ff_expansion),
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
    split: Any,
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


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _plot_history(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["train_macro_f1"], label="train")
    axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_final_train_val_history(history_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history_df["epoch"], history_df["train_val_loss"], label="train+val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_val_accuracy"], label="train+val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["train_val_macro_f1"], label="train+val")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    output_path: Path,
    title: str,
) -> None:
    cm = np.asarray(matrix, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    max_value = float(cm.max()) if cm.size else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _ordered_label_names_after_mapping(labels: Sequence[str]) -> tuple[str, ...]:
    labels_set = {str(label) for label in labels}
    ordered: list[str] = []
    for label in ROLE_LABELS_V1:
        mapped_label = WINGBACK_TO_FULLBACK_LABEL_MAP.get(label, label)
        if mapped_label in labels_set and mapped_label not in ordered:
            ordered.append(mapped_label)
    extras = sorted(labels_set.difference(ordered))
    return tuple(ordered + extras)


def _remap_dataset_labels(
    *,
    dataset: DatasetBundle,
    label_map: Mapping[str, str] | None,
) -> tuple[DatasetBundle, dict[str, Any]]:
    normalized_map = {
        str(source): str(target)
        for source, target in (label_map or {}).items()
        if str(source) != str(target)
    }
    if not normalized_map:
        return dataset, {
            "enabled": False,
            "map": {},
            "before_counts": {
                str(label): int(count)
                for label, count in dataset.samples_df["label"].astype(str).value_counts().sort_index().items()
            },
            "after_counts": {
                str(label): int(count)
                for label, count in dataset.samples_df["label"].astype(str).value_counts().sort_index().items()
            },
        }

    samples_df = dataset.samples_df.copy()
    before_counts = {
        str(label): int(count)
        for label, count in samples_df["label"].astype(str).value_counts().sort_index().items()
    }
    samples_df["label"] = samples_df["label"].astype(str).replace(normalized_map)
    if "role_label" in samples_df.columns:
        samples_df["role_label"] = samples_df["role_label"].astype(str).replace(normalized_map)

    after_counts = {
        str(label): int(count)
        for label, count in samples_df["label"].astype(str).value_counts().sort_index().items()
    }
    label_names = _ordered_label_names_after_mapping(samples_df["label"].astype(str).tolist())

    remapped_dataset = DatasetBundle(
        samples_df=samples_df.reset_index(drop=True),
        objective=dataset.objective,
        teammates=dataset.teammates,
        teammate_mask=dataset.teammate_mask,
        feature_spec=dataset.feature_spec,
        label_names=label_names,
    )
    return remapped_dataset, {
        "enabled": True,
        "map": normalized_map,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "before_label_names": list(dataset.label_names),
        "after_label_names": list(label_names),
    }


def _assert_no_wingback_labels_when_merged(dataset: DatasetBundle) -> None:
    forbidden_labels = set(WINGBACK_TO_FULLBACK_LABEL_MAP.keys())
    label_names = set(str(label) for label in dataset.label_names)
    remaining_in_label_names = sorted(label_names.intersection(forbidden_labels))
    remaining_in_samples = sorted(
        set(dataset.samples_df["label"].astype(str).unique().tolist()).intersection(forbidden_labels)
    )
    if remaining_in_label_names or remaining_in_samples:
        raise RuntimeError(
            "El remapeo de carrileros no se aplicó correctamente antes del entrenamiento. "
            f"Quedan en label_names={remaining_in_label_names} y en samples={remaining_in_samples}."
        )


def _prepare_dataset(
    project_root: Path,
    base_table_path: Path | None,
    max_teammates: int,
    seed: int,
    prefer_cached_common_dataset: bool,
    label_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    dataset = load_labeled_dataset_from_base_table(
        project_root=project_root,
        base_table_path=base_table_path,
        max_teammates=int(max_teammates),
        prefer_cached_common_dataset=bool(prefer_cached_common_dataset),
    )
    dataset, label_mapping_payload = _remap_dataset_labels(dataset=dataset, label_map=label_map)
    if label_mapping_payload["enabled"]:
        _assert_no_wingback_labels_when_merged(dataset)
    split = split_dataset_by_match(dataset, seed=seed)

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

    return {
        "dataset": dataset,
        "split": split,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "labels_idx": labels_idx,
        "standardizer": standardizer,
        "objective_scaled": objective_scaled,
        "teammates_scaled": teammates_scaled,
        "label_mapping": label_mapping_payload,
    }


def train_single_experiment(
    *,
    prepared: Mapping[str, Any],
    config: TrainingConfig,
    run_dir: Path,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    prefer_cached_common_dataset: bool = True,
) -> dict[str, Any]:
    set_global_seed(config.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = prepared["dataset"]
    split = prepared["split"]
    labels_idx = prepared["labels_idx"]
    standardizer: FeatureStandardizer = prepared["standardizer"]
    label_mapping = dict(prepared.get("label_mapping", {"enabled": False, "map": {}}))

    train_loader, val_loader, test_loader, used_indices = _build_dataloaders(
        objective_scaled=prepared["objective_scaled"],
        teammates_scaled=prepared["teammates_scaled"],
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
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=float(config.label_smoothing),
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

    checkpoint_path = run_dir / "best_model.pt"
    history_rows: list[dict[str, Any]] = []
    best_val_macro_f1 = -np.inf
    best_val_loss = math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    no_f1_improve = 0
    val_loss_worse_count = 0
    stop_reason = "max_epochs"

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

        current_val_loss = float(val_epoch["loss"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_epoch["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": current_val_loss,
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "generalization_gap_loss": current_val_loss - float(train_epoch["loss"]),
        }
        history_rows.append(row)

        f1_improved = float(val_metrics["macro_f1"]) > float(best_val_macro_f1)
        if f1_improved:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)
            no_f1_improve = 0
        else:
            no_f1_improve += 1

        if current_val_loss < best_val_loss - 1e-6:
            best_val_loss = current_val_loss
            val_loss_worse_count = 0
        else:
            val_loss_worse_count += 1

        if best_state is not None and f1_improved:
            checkpoint = {
                "created_at": datetime.now().isoformat(),
                "model_state_dict": best_state,
                "training_config": asdict(config),
                "objective_feature_names": list(dataset.feature_spec.objective_feature_names),
                "teammate_feature_names": list(dataset.feature_spec.teammate_feature_names),
                "label_names": list(dataset.label_names),
                "label_to_idx": prepared["label_to_idx"],
                "idx_to_label": prepared["idx_to_label"],
                "standardizer": standardizer.to_state(),
                "best_epoch": int(best_epoch),
                "used_cached_common_dataset": bool(prefer_cached_common_dataset),
                "label_mapping": label_mapping,
                "heuristics": {
                    "goalkeeper_role": "POR",
                },
                "experiment_note": (
                    "Checkpoint experimental generado por "
                    "experiments.positions.set_transformer_grid_search."
                ),
            }
            torch.save(checkpoint, checkpoint_path)

        if no_f1_improve >= int(config.patience):
            stop_reason = f"early_stopping_val_macro_f1_patience_{config.patience}"
            break
        if val_loss_worse_count >= int(config.overfit_patience):
            stop_reason = f"early_stopping_val_loss_overfit_patience_{config.overfit_patience}"
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

    metrics_payload = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "stop_reason": stop_reason,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1_during_training": float(best_val_macro_f1),
        "best_val_loss_during_training": float(best_val_loss),
        "num_classes_trained": int(len(dataset.label_names)),
        "trained_labels": list(dataset.label_names),
        "missing_known_labels": [
            label for label in ROLE_LABELS_V1 if label not in set(dataset.label_names)
        ],
        "used_cached_common_dataset": bool(prefer_cached_common_dataset),
        "label_mapping": label_mapping,
        "train": _classification_metrics(
            y_true=train_final["y_true"],
            y_pred=train_final["y_pred"],
            label_names=dataset.label_names,
        ),
        "val": _classification_metrics(
            y_true=val_final["y_true"],
            y_pred=val_final["y_pred"],
            label_names=dataset.label_names,
        ),
        "test": _classification_metrics(
            y_true=test_final["y_true"],
            y_pred=test_final["y_pred"],
            label_names=dataset.label_names,
        ),
    }
    split_payload = {
        "train_matches": list(split.train_matches),
        "val_matches": list(split.val_matches),
        "test_matches": list(split.test_matches),
        "num_train_samples": int(len(used_indices["train_idx"])),
        "num_val_samples": int(len(used_indices["val_idx"])),
        "num_test_samples": int(len(used_indices["test_idx"])),
    }

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    _save_json(run_dir / "config.json", asdict(config))
    _save_json(run_dir / "metrics.json", metrics_payload)
    _save_json(run_dir / "split.json", split_payload)
    _plot_history(history_df, run_dir / "training_curves.png")
    _plot_confusion_matrix(
        matrix=metrics_payload["val"]["confusion_matrix"],
        labels=dataset.label_names,
        output_path=run_dir / "confusion_matrix_val.png",
        title="Matriz de confusión en validación",
    )
    _plot_confusion_matrix(
        matrix=metrics_payload["test"]["confusion_matrix"],
        labels=dataset.label_names,
        output_path=run_dir / "confusion_matrix_test.png",
        title="Matriz de confusión en test",
    )

    return {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "config": asdict(config),
        "history_df": history_df,
        "metrics": metrics_payload,
        "split": split_payload,
    }


def _load_final_training_config(
    *,
    best_hyperparameters_path: Path,
    project_root: Path,
    final_epochs: int | None,
) -> tuple[TrainingConfig, dict[str, Any]]:
    path = best_hyperparameters_path
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de hiperparámetros: {path}")

    with path.open("r", encoding="utf-8") as f:
        best_payload = json.load(f)

    epochs = int(final_epochs if final_epochs is not None else best_payload["best_epoch"])
    if epochs < 1:
        raise ValueError(f"final_epochs debe ser >= 1, recibido: {epochs}")

    config = TrainingConfig(
        seed=int(best_payload.get("seed", 42)),
        epochs=epochs,
        batch_size=int(best_payload.get("batch_size", 512)),
        learning_rate=float(best_payload["learning_rate"]),
        weight_decay=float(best_payload["weight_decay"]),
        dropout=float(best_payload["dropout"]),
        label_smoothing=float(best_payload.get("label_smoothing", 0.0)),
        teammate_embed_dim=int(best_payload["teammate_embed_dim"]),
        objective_hidden_dim=int(best_payload["objective_hidden_dim"]),
        objective_num_layers=int(best_payload["objective_num_layers"]),
        set_hidden_dim=int(best_payload["set_hidden_dim"]),
        fusion_hidden_dim=int(best_payload["fusion_hidden_dim"]),
        num_heads=int(best_payload.get("num_heads", 4)),
        num_set_blocks=int(best_payload["num_set_blocks"]),
        ff_expansion=int(best_payload["ff_expansion"]),
        patience=0,
        overfit_patience=0,
        experiment_group="final_train_val",
    )
    return config, {
        "path": str(path),
        "payload": best_payload,
        "selected_epoch_source": "best_epoch" if final_epochs is None else "manual_override",
    }


def train_final_train_val_model(
    *,
    project_root: Path | None = None,
    base_table_path: Path | None = None,
    output_dir: Path | None = None,
    best_hyperparameters_path: Path = DEFAULT_BEST_HYPERPARAMETERS_PATH,
    final_epochs: int | None = None,
    prefer_cached_common_dataset: bool = True,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    output_root = project_root / (output_dir or DEFAULT_FINAL_TRAIN_VAL_OUTPUT_DIR)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config, best_hparams_source = _load_final_training_config(
        best_hyperparameters_path=best_hyperparameters_path,
        project_root=project_root,
        final_epochs=final_epochs,
    )
    set_global_seed(config.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    print(f"[final] Preparando dataset desde {base_table_path or DEFAULT_BASE_TABLE_PATH}...")
    prepared = _prepare_dataset(
        project_root=project_root,
        base_table_path=base_table_path,
        max_teammates=DEFAULT_MAX_TEAMMATES,
        seed=int(config.seed),
        prefer_cached_common_dataset=prefer_cached_common_dataset,
        label_map=WINGBACK_TO_FULLBACK_LABEL_MAP,
    )

    dataset = prepared["dataset"]
    split = prepared["split"]
    labels_idx = prepared["labels_idx"]
    label_mapping = dict(prepared.get("label_mapping", {"enabled": False, "map": {}}))

    train_val_idx = np.sort(
        np.concatenate(
            [
                np.asarray(split.train_idx, dtype=np.int64),
                np.asarray(split.val_idx, dtype=np.int64),
            ]
        )
    ).astype(np.int64)
    train_val_idx = _subset_indices_keep_labels(
        train_val_idx,
        max_train_samples,
        seed=int(config.seed),
        labels_idx=labels_idx,
    )
    test_idx = _subset_indices(
        split.test_idx,
        max_test_samples,
        seed=int(config.seed) + 2,
    )

    standardizer = FeatureStandardizer.fit(
        objective=dataset.objective[train_val_idx],
        teammates=dataset.teammates[train_val_idx],
        teammate_mask=dataset.teammate_mask[train_val_idx],
    )
    objective_scaled, teammates_scaled = standardizer.transform(
        objective=dataset.objective,
        teammates=dataset.teammates,
        teammate_mask=dataset.teammate_mask,
    )

    train_val_loader = DataLoader(
        RoleDataset(
            objective=objective_scaled[train_val_idx],
            teammates=teammates_scaled[train_val_idx],
            teammate_mask=dataset.teammate_mask[train_val_idx],
            labels=labels_idx[train_val_idx],
        ),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        drop_last=False,
    )
    train_val_eval_loader = DataLoader(
        RoleDataset(
            objective=objective_scaled[train_val_idx],
            teammates=teammates_scaled[train_val_idx],
            teammate_mask=dataset.teammate_mask[train_val_idx],
            labels=labels_idx[train_val_idx],
        ),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        drop_last=False,
    )
    test_loader = DataLoader(
        RoleDataset(
            objective=objective_scaled[test_idx],
            teammates=teammates_scaled[test_idx],
            teammate_mask=dataset.teammate_mask[test_idx],
            labels=labels_idx[test_idx],
        ),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        drop_last=False,
    )

    device = _select_device()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(dataset.label_names)),
        y=labels_idx[train_val_idx],
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=float(config.label_smoothing),
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

    print(
        "[final] Entrenando con train+val "
        f"({len(train_val_idx)} muestras) durante {config.epochs} epoch(s). "
        "El test queda reservado para la evaluación final."
    )
    history_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
        train_epoch = _run_epoch(
            model=model,
            loader=train_val_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        train_metrics = _classification_metrics(
            y_true=train_epoch["y_true"],
            y_pred=train_epoch["y_pred"],
            label_names=dataset.label_names,
        )
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_val_loss": float(train_epoch["loss"]),
                "train_val_accuracy": float(train_metrics["accuracy"]),
                "train_val_macro_f1": float(train_metrics["macro_f1"]),
                "train_val_weighted_f1": float(train_metrics["weighted_f1"]),
            }
        )
        print(
            "[final] "
            f"epoch={epoch}/{config.epochs} "
            f"loss={float(train_epoch['loss']):.4f} "
            f"acc={float(train_metrics['accuracy']):.4f} "
            f"macro_f1={float(train_metrics['macro_f1']):.4f}"
        )

    train_val_final = _run_epoch(
        model=model,
        loader=train_val_eval_loader,
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
    train_val_metrics = _classification_metrics(
        y_true=train_val_final["y_true"],
        y_pred=train_val_final["y_pred"],
        label_names=dataset.label_names,
    )
    test_metrics = _classification_metrics(
        y_true=test_final["y_true"],
        y_pred=test_final["y_pred"],
        label_names=dataset.label_names,
    )

    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = {
        "created_at": datetime.now().isoformat(),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "training_config": asdict(config),
        "objective_feature_names": list(dataset.feature_spec.objective_feature_names),
        "teammate_feature_names": list(dataset.feature_spec.teammate_feature_names),
        "label_names": list(dataset.label_names),
        "label_to_idx": prepared["label_to_idx"],
        "idx_to_label": prepared["idx_to_label"],
        "standardizer": standardizer.to_state(),
        "best_epoch": int(config.epochs),
        "used_cached_common_dataset": bool(prefer_cached_common_dataset),
        "label_mapping": label_mapping,
        "heuristics": {
            "goalkeeper_role": "POR",
        },
        "training_mode": "final_train_val_fixed_epoch",
        "experiment_note": (
            "Checkpoint final entrenado con train+val usando los hiperparámetros "
            "seleccionados por validación. El conjunto de test solo se usa al final."
        ),
    }
    torch.save(checkpoint, checkpoint_path)

    split_payload = {
        "original_train_matches": list(split.train_matches),
        "original_val_matches": list(split.val_matches),
        "test_matches": list(split.test_matches),
        "train_val_matches": sorted(set(split.train_matches).union(split.val_matches)),
        "num_original_train_samples": int(len(split.train_idx)),
        "num_original_val_samples": int(len(split.val_idx)),
        "num_train_val_samples": int(len(train_val_idx)),
        "num_test_samples": int(len(test_idx)),
    }
    metrics_payload = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "training_mode": "final_train_val_fixed_epoch",
        "source_best_hyperparameters": best_hparams_source,
        "num_classes_trained": int(len(dataset.label_names)),
        "trained_labels": list(dataset.label_names),
        "used_cached_common_dataset": bool(prefer_cached_common_dataset),
        "label_mapping": label_mapping,
        "train_val": train_val_metrics,
        "test": test_metrics,
    }
    dataset_summary = {
        "num_samples": int(len(dataset.samples_df)),
        "objective_dim": int(dataset.objective.shape[1]),
        "teammate_shape": list(dataset.teammates.shape),
        "label_names": list(dataset.label_names),
        "label_counts": {
            str(label): int(count)
            for label, count in dataset.samples_df["label"].astype(str).value_counts().sort_index().items()
        },
        "label_mapping": label_mapping,
    }

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    _save_json(run_dir / "config.json", asdict(config))
    _save_json(run_dir / "metrics.json", metrics_payload)
    _save_json(run_dir / "split.json", split_payload)
    _save_json(run_dir / "dataset_summary.json", dataset_summary)
    _save_json(run_dir / "source_best_hyperparameters.json", best_hparams_source)
    _plot_final_train_val_history(history_df, run_dir / "training_curves.png")
    _plot_confusion_matrix(
        matrix=train_val_metrics["confusion_matrix"],
        labels=dataset.label_names,
        output_path=run_dir / "confusion_matrix_train_val.png",
        title="Matriz de confusión en train+val",
    )
    _plot_confusion_matrix(
        matrix=test_metrics["confusion_matrix"],
        labels=dataset.label_names,
        output_path=run_dir / "confusion_matrix_test.png",
        title="Matriz de confusión en test",
    )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "training_mode": "final_train_val_fixed_epoch",
        "epochs": int(config.epochs),
        "train_val_accuracy": float(train_val_metrics["accuracy"]),
        "train_val_macro_f1": float(train_val_metrics["macro_f1"]),
        "train_val_weighted_f1": float(train_val_metrics["weighted_f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
    }
    _save_json(run_dir / "summary.json", summary)
    print("\n[final] Resultado final:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[final] Artefactos guardados en: {run_dir}")
    return {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "config": asdict(config),
        "history_df": history_df,
        "metrics": metrics_payload,
        "split": split_payload,
        "summary": summary,
    }


def _run_name(index: int, config: TrainingConfig) -> str:
    group = str(config.experiment_group).replace("/", "_").replace(" ", "_")
    return (
        f"run_{index:03d}"
        f"_{group}"
        f"_h{config.set_hidden_dim}"
        f"_objh{config.objective_hidden_dim}"
        f"_te{config.teammate_embed_dim}"
        f"_fu{config.fusion_hidden_dim}"
        f"_obj{config.objective_num_layers}"
        f"_ff{config.ff_expansion}"
        f"_blocks{config.num_set_blocks}"
        f"_drop{str(config.dropout).replace('.', 'p')}"
        f"_ls{str(config.label_smoothing).replace('.', 'p')}"
        f"_lr{str(config.learning_rate).replace('.', 'p').replace('-', 'm')}"
        f"_wd{str(config.weight_decay).replace('.', 'p').replace('-', 'm')}"
    )


def _summary_row(result: Mapping[str, Any]) -> dict[str, Any]:
    config = result["config"]
    metrics = result["metrics"]
    return {
        "run_dir": str(result["run_dir"]),
        "checkpoint_path": str(result["checkpoint_path"]),
        "teammate_embed_dim": int(config["teammate_embed_dim"]),
        "objective_hidden_dim": int(config["objective_hidden_dim"]),
        "set_hidden_dim": int(config["set_hidden_dim"]),
        "fusion_hidden_dim": int(config["fusion_hidden_dim"]),
        "experiment_group": str(config.get("experiment_group", "default")),
        "objective_num_layers": int(config["objective_num_layers"]),
        "ff_expansion": int(config["ff_expansion"]),
        "num_set_blocks": int(config["num_set_blocks"]),
        "dropout": float(config["dropout"]),
        "label_smoothing": float(config.get("label_smoothing", 0.0)),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "label_mapping_enabled": bool(metrics.get("label_mapping", {}).get("enabled", False)),
        "best_epoch": int(metrics["best_epoch"]),
        "stop_reason": str(metrics["stop_reason"]),
        "val_accuracy": float(metrics["val"]["accuracy"]),
        "val_macro_f1": float(metrics["val"]["macro_f1"]),
        "val_weighted_f1": float(metrics["val"]["weighted_f1"]),
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_macro_f1": float(metrics["test"]["macro_f1"]),
        "test_weighted_f1": float(metrics["test"]["weighted_f1"]),
        "best_val_loss": float(metrics["best_val_loss_during_training"]),
        "best_val_macro_f1_during_training": float(
            metrics["best_val_macro_f1_during_training"]
        ),
    }


def _simple_grid_configs(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[TrainingConfig], dict[str, Any]]:
    grid_values = {
        "objective_num_layers": [1, 2, 3],
        "ff_expansion": [2, 4],
        "num_set_blocks": [2, 3, 4],
        "dropout": [0.10, 0.15, 0.20],
    }
    configs: list[TrainingConfig] = []
    for values in itertools.product(*grid_values.values()):
        params = dict(zip(grid_values.keys(), values, strict=True))
        configs.append(
            TrainingConfig(
                seed=int(seed),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                objective_num_layers=int(params["objective_num_layers"]),
                ff_expansion=int(params["ff_expansion"]),
                num_set_blocks=int(params["num_set_blocks"]),
                dropout=float(params["dropout"]),
                patience=3,
                overfit_patience=3,
            )
        )

    definition = {
        "preset": "simple",
        "grid": grid_values,
        "fixed": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": 3,
            "overfit_patience": 3,
            "teammate_embed_dim": 64,
            "objective_hidden_dim": 128,
            "set_hidden_dim": 128,
            "fusion_hidden_dim": 128,
            "num_heads": 4,
        },
        "num_runs": len(configs),
    }
    return configs, definition


def _full_grid_configs(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[list[TrainingConfig], dict[str, Any]]:
    # Este preset está guiado por el primer grid:
    # - los mejores resultados aparecieron con 3-4 bloques, así que exploramos 3-5;
    # - FF x2 fue mejor de media, pero se conserva x4 como alternativa expresiva;
    # - el overfitting apareció muy pronto, por eso se prueban más dropout y weight decay;
    # - se amplía la anchura del modelo con perfiles coherentes en lugar de cruzar
    #   todas las dimensiones de forma explosiva.
    width_profiles = [
        {
            "name": "base_128",
            "teammate_embed_dim": 64,
            "objective_hidden_dim": 128,
            "set_hidden_dim": 128,
            "fusion_hidden_dim": 128,
        },
        {
            "name": "wide_192",
            "teammate_embed_dim": 128,
            "objective_hidden_dim": 256,
            "set_hidden_dim": 192,
            "fusion_hidden_dim": 256,
        },
        {
            "name": "wide_256",
            "teammate_embed_dim": 128,
            "objective_hidden_dim": 256,
            "set_hidden_dim": 256,
            "fusion_hidden_dim": 256,
        },
        {
            "name": "wide_256_obj384",
            "teammate_embed_dim": 256,
            "objective_hidden_dim": 384,
            "set_hidden_dim": 256,
            "fusion_hidden_dim": 384,
        },
    ]
    regularization_profiles = [
        {"name": "drop010_wd1e4", "dropout": 0.10, "weight_decay": 1e-4},
        {"name": "drop015_wd1e4", "dropout": 0.15, "weight_decay": 1e-4},
        {"name": "drop020_wd5e4", "dropout": 0.20, "weight_decay": 5e-4},
        {"name": "drop025_wd5e4", "dropout": 0.25, "weight_decay": 5e-4},
        {"name": "drop030_wd1e3", "dropout": 0.30, "weight_decay": 1e-3},
    ]
    objective_num_layers = [1, 2]
    ff_expansion = [2, 4]
    num_set_blocks = [3, 4, 5]
    learning_rates = [1e-3, 5e-4]

    configs: list[TrainingConfig] = []
    for width, reg, obj_layers, ff, blocks, lr in itertools.product(
        width_profiles,
        regularization_profiles,
        objective_num_layers,
        ff_expansion,
        num_set_blocks,
        learning_rates,
    ):
        configs.append(
            TrainingConfig(
                seed=int(seed),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(lr),
                weight_decay=float(reg["weight_decay"]),
                dropout=float(reg["dropout"]),
                teammate_embed_dim=int(width["teammate_embed_dim"]),
                objective_hidden_dim=int(width["objective_hidden_dim"]),
                set_hidden_dim=int(width["set_hidden_dim"]),
                fusion_hidden_dim=int(width["fusion_hidden_dim"]),
                objective_num_layers=int(obj_layers),
                ff_expansion=int(ff),
                num_set_blocks=int(blocks),
                patience=3,
                overfit_patience=3,
            )
        )

    definition = {
        "preset": "full",
        "search_strategy": (
            "Grid completo focalizado tras el primer experimento: amplia anchura, "
            "regularización y bloques alrededor de las configuraciones con mejor val_macro_f1."
        ),
        "width_profiles": width_profiles,
        "regularization_profiles": regularization_profiles,
        "grid": {
            "objective_num_layers": objective_num_layers,
            "ff_expansion": ff_expansion,
            "num_set_blocks": num_set_blocks,
            "learning_rate": learning_rates,
        },
        "fixed": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": 3,
            "overfit_patience": 3,
            "num_heads": 4,
        },
        "num_runs": len(configs),
    }
    return configs, definition


def _best_run_grid_configs(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[list[TrainingConfig], dict[str, Any]]:
    config = TrainingConfig(
        seed=int(seed),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.10,
        teammate_embed_dim=256,
        objective_hidden_dim=384,
        objective_num_layers=2,
        set_hidden_dim=256,
        fusion_hidden_dim=384,
        num_heads=4,
        num_set_blocks=3,
        ff_expansion=4,
        patience=3,
        overfit_patience=3,
    )
    return [config], {
        "preset": "best-run",
        "source": "experiments/positions/grid_search_outputs/20260426_155024/runs/best_run/config.json",
        "search_strategy": (
            "Reentrena una única configuración con los hiperparámetros del mejor "
            "modelo encontrado en el grid ampliado."
        ),
        "fixed": asdict(config),
        "num_runs": 1,
    }


def _configs_from_base_and_regularization_grid(
    *,
    base: Mapping[str, Any],
    learning_rates: Sequence[float],
    weight_decays: Sequence[float],
    dropouts: Sequence[float],
    label_smoothing_values: Sequence[float],
    seed: int,
    epochs: int,
    batch_size: int,
) -> list[TrainingConfig]:
    configs: list[TrainingConfig] = []
    for lr, wd, dropout, label_smoothing in itertools.product(
        learning_rates,
        weight_decays,
        dropouts,
        label_smoothing_values,
    ):
        configs.append(
            TrainingConfig(
                seed=int(seed),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(lr),
                weight_decay=float(wd),
                dropout=float(dropout),
                label_smoothing=float(label_smoothing),
                teammate_embed_dim=int(base["teammate_embed_dim"]),
                objective_hidden_dim=int(base["objective_hidden_dim"]),
                objective_num_layers=int(base["objective_num_layers"]),
                set_hidden_dim=int(base["set_hidden_dim"]),
                fusion_hidden_dim=int(base["fusion_hidden_dim"]),
                num_heads=int(base.get("num_heads", 4)),
                num_set_blocks=int(base["num_set_blocks"]),
                ff_expansion=int(base["ff_expansion"]),
                patience=3,
                overfit_patience=3,
                experiment_group=str(base["name"]),
            )
        )
    return configs


def _refine_top3_grid_configs(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[list[TrainingConfig], dict[str, Any]]:
    # Top 3 por val_macro_f1 del grid corregido 20260426_195856. Cada subgrid
    # mantiene fija la arquitectura y prueba solo valores nuevos de regularización
    # y learning rate, para no repetir el barrido anterior.
    subgrids = [
        {
            "name": "refine_run120_top_val",
            "source_run": "run_120_h128_objh128_te64_fu128_obj2_ff4_blocks5_drop0p3_lr0p0005_wd0p001",
            "teammate_embed_dim": 64,
            "objective_hidden_dim": 128,
            "set_hidden_dim": 128,
            "fusion_hidden_dim": 128,
            "objective_num_layers": 2,
            "num_set_blocks": 5,
            "ff_expansion": 4,
            "learning_rates": [3e-4, 2e-4, 1e-4],
            "weight_decays": [1.5e-3, 2e-3],
            "dropouts": [0.27, 0.33, 0.35],
            "label_smoothing": [0.05, 0.10],
        },
        {
            "name": "refine_run342_second_val",
            "source_run": "run_342_h256_objh256_te128_fu256_obj1_ff2_blocks5_drop0p3_lr0p0005_wd0p001",
            "teammate_embed_dim": 128,
            "objective_hidden_dim": 256,
            "set_hidden_dim": 256,
            "fusion_hidden_dim": 256,
            "objective_num_layers": 1,
            "num_set_blocks": 5,
            "ff_expansion": 2,
            "learning_rates": [3e-4, 2e-4, 1e-4],
            "weight_decays": [1.5e-3, 2e-3],
            "dropouts": [0.27, 0.33, 0.35],
            "label_smoothing": [0.05, 0.10],
        },
        {
            "name": "refine_run290_third_val_best_balance",
            "source_run": "run_290_h256_objh256_te128_fu256_obj1_ff2_blocks3_drop0p2_lr0p0005_wd0p0005",
            "teammate_embed_dim": 128,
            "objective_hidden_dim": 256,
            "set_hidden_dim": 256,
            "fusion_hidden_dim": 256,
            "objective_num_layers": 1,
            "num_set_blocks": 3,
            "ff_expansion": 2,
            "learning_rates": [3e-4, 2e-4, 1e-4],
            "weight_decays": [7.5e-4, 1.5e-3],
            "dropouts": [0.18, 0.22, 0.27],
            "label_smoothing": [0.05, 0.10],
        },
    ]

    configs: list[TrainingConfig] = []
    for subgrid in subgrids:
        configs.extend(
            _configs_from_base_and_regularization_grid(
                base=subgrid,
                learning_rates=subgrid["learning_rates"],
                weight_decays=subgrid["weight_decays"],
                dropouts=subgrid["dropouts"],
                label_smoothing_values=subgrid["label_smoothing"],
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
            )
        )

    return configs, {
        "preset": "refine-top3",
        "source_grid": "experiments/positions/grid_search_outputs/20260426_195856/grid_search_summary.csv",
        "search_strategy": (
            "Tres subgrids secuenciales alrededor de los tres mejores modelos por "
            "val_macro_f1. La arquitectura se mantiene fija en cada familia y se "
            "prueban valores nuevos de learning rate, weight decay, dropout y "
            "label smoothing."
        ),
        "subgrids": subgrids,
        "fixed": {
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": 3,
            "overfit_patience": 3,
            "num_heads": 4,
        },
        "num_runs": len(configs),
    }


def _build_grid_configs(
    *,
    preset: str,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[TrainingConfig], dict[str, Any]]:
    if preset == "simple":
        return _simple_grid_configs(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    if preset == "full":
        return _full_grid_configs(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
    if preset == "best-run":
        return _best_run_grid_configs(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
    if preset == "refine-top3":
        return _refine_top3_grid_configs(
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
        )
    raise ValueError(f"Preset de grid no soportado: {preset}")


def run_grid_search(
    *,
    project_root: Path | None = None,
    base_table_path: Path | None = None,
    output_dir: Path | None = None,
    grid_preset: str = "simple",
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    prefer_cached_common_dataset: bool = True,
    merge_wingbacks: bool = False,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    output_root = project_root / (output_dir or DEFAULT_OUTPUT_DIR)
    search_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = output_root / search_id
    runs_dir = search_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    configs, grid_definition = _build_grid_configs(
        preset=str(grid_preset),
        seed=int(seed),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    label_map = WINGBACK_TO_FULLBACK_LABEL_MAP if merge_wingbacks else None
    grid_definition["label_mapping"] = {
        "enabled": bool(merge_wingbacks),
        "map": dict(label_map or {}),
    }
    _save_json(search_dir / "grid_definition.json", grid_definition)

    print(f"[grid] Preparando dataset desde {base_table_path or DEFAULT_BASE_TABLE_PATH}...")
    prepared = _prepare_dataset(
        project_root=project_root,
        base_table_path=base_table_path,
        max_teammates=DEFAULT_MAX_TEAMMATES,
        seed=seed,
        prefer_cached_common_dataset=prefer_cached_common_dataset,
        label_map=label_map,
    )
    dataset = prepared["dataset"]
    dataset_summary = {
        "num_samples": int(len(dataset.samples_df)),
        "objective_dim": int(dataset.objective.shape[1]),
        "teammate_shape": list(dataset.teammates.shape),
        "label_names": list(dataset.label_names),
        "label_counts": {
            str(label): int(count)
            for label, count in dataset.samples_df["label"].astype(str).value_counts().sort_index().items()
        },
        "label_mapping": prepared["label_mapping"],
    }
    _save_json(search_dir / "dataset_summary.json", dataset_summary)

    summary_rows: list[dict[str, Any]] = []
    total_runs = len(configs)
    best_row: dict[str, Any] | None = None

    for index, config in enumerate(configs, start=1):
        run_dir = runs_dir / _run_name(index=index, config=config)
        print(
            "[grid] "
            f"{index}/{total_runs} "
            f"h={config.set_hidden_dim} "
            f"obj_h={config.objective_hidden_dim} "
            f"te={config.teammate_embed_dim} "
            f"fu={config.fusion_hidden_dim} "
            f"objective_layers={config.objective_num_layers} "
            f"ff=x{config.ff_expansion} "
            f"blocks={config.num_set_blocks} "
            f"dropout={config.dropout:.2f} "
            f"lr={config.learning_rate:g} "
            f"wd={config.weight_decay:g}"
        )
        result = train_single_experiment(
            prepared=prepared,
            config=config,
            run_dir=run_dir,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            prefer_cached_common_dataset=prefer_cached_common_dataset,
        )
        row = _summary_row(result)
        summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["val_macro_f1", "val_accuracy", "test_macro_f1"],
            ascending=[False, False, False],
        )
        summary_df.to_csv(search_dir / "grid_search_summary.csv", index=False)
        _save_json(
            search_dir / "grid_search_summary.json",
            {"runs": summary_df.to_dict(orient="records")},
        )

        if best_row is None or row["val_macro_f1"] > best_row["val_macro_f1"]:
            best_row = row
            shutil.copy2(Path(row["checkpoint_path"]), search_dir / "best_model.pt")
            _save_json(search_dir / "best_hyperparameters.json", row)

    if best_row is None:
        raise RuntimeError("El grid search no produjo resultados.")

    best_run_source = Path(best_row["run_dir"])
    best_run_copy = runs_dir / "best_run"
    if best_run_source.exists():
        if best_run_copy.exists():
            shutil.rmtree(best_run_copy)
        shutil.copytree(best_run_source, best_run_copy)

    print("\n[grid] Mejor configuración por val_macro_f1:")
    print(json.dumps(best_row, ensure_ascii=False, indent=2))
    print(f"[grid] Resultados guardados en: {search_dir}")

    return {
        "search_dir": search_dir,
        "best": best_row,
        "summary": pd.DataFrame(summary_rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grid search para optimizar el Set Transformer de posiciones. "
            "Por defecto, los resultados se guardan en models/positions/grid_search_outputs. "
            "Con --final-train-val se entrena el modelo final usando train+val y se evalúa en test."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--base-table-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--final-train-val",
        action="store_true",
        help=(
            "Entrena un único modelo final con train+val usando los mejores hiperparámetros "
            "guardados y evalúa una sola vez en test."
        ),
    )
    parser.add_argument(
        "--best-hyperparameters-path",
        type=Path,
        default=DEFAULT_BEST_HYPERPARAMETERS_PATH,
        help="JSON con los hiperparámetros seleccionados por validación.",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=None,
        help=(
            "Epochs para el entrenamiento final. Si no se indica, usa best_epoch del JSON "
            "de hiperparámetros."
        ),
    )
    parser.add_argument(
        "--grid-preset",
        choices=("simple", "full", "best-run", "refine-top3"),
        default="simple",
        help=(
            "simple reproduce el primer grid; full amplía anchura y regularización; "
            "best-run reentrena solo la mejor configuración conocida; refine-top3 "
            "afina los tres mejores modelos del grid corregido."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--rebuild-from-base-table",
        action="store_true",
        help="Ignora el dataset común cacheado y reconstruye samples desde base_table.csv.",
    )
    parser.add_argument(
        "--merge-wingbacks",
        action="store_true",
        help="Fusiona CI->LI y CD->LD antes de entrenar/evaluar el modelo experimental.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if bool(args.final_train_val):
        output_dir = (
            DEFAULT_FINAL_TRAIN_VAL_OUTPUT_DIR
            if args.output_dir == DEFAULT_OUTPUT_DIR
            else args.output_dir
        )
        train_final_train_val_model(
            project_root=args.project_root,
            base_table_path=args.base_table_path,
            output_dir=output_dir,
            best_hyperparameters_path=args.best_hyperparameters_path,
            final_epochs=args.final_epochs,
            prefer_cached_common_dataset=not bool(args.rebuild_from_base_table),
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
        )
        return 0

    run_grid_search(
        project_root=args.project_root,
        base_table_path=args.base_table_path,
        output_dir=args.output_dir,
        grid_preset=str(args.grid_preset),
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        prefer_cached_common_dataset=not bool(args.rebuild_from_base_table),
        merge_wingbacks=bool(args.merge_wingbacks),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
