from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from ..data import ROLE_LABELS_V1, find_project_root
from .architecture import RoleSetTransformer
from .config import ArtifactPaths, TrainingConfig
from .data_utils import (
    FeatureStandardizer,
    build_dataloaders,
    load_labeled_dataset_from_base_table,
    set_global_seed,
    split_dataset_by_match,
)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    label_to_idx = {label: idx for idx, label in enumerate(dataset.label_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
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
    train_loader, val_loader, test_loader, used_indices = build_dataloaders(
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

    device = select_device()
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

    history_rows: list[dict[str, Any]] = []
    best_val_macro_f1 = -np.inf
    best_val_loss = math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    no_improve = 0
    overfit_count = 0
    stop_reason = "max_epochs"
    for epoch in range(1, int(config.epochs) + 1):
        train_epoch = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_epoch = _run_epoch(model, val_loader, criterion, device, None)
        train_metrics = _classification_metrics(train_epoch["y_true"], train_epoch["y_pred"], dataset.label_names)
        val_metrics = _classification_metrics(val_epoch["y_true"], val_epoch["y_pred"], dataset.label_names)
        train_loss = float(train_epoch["loss"])
        val_loss = float(val_epoch["loss"])
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_loss": val_loss,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "generalization_gap_loss": float(val_loss - train_loss),
            }
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            overfit_count = 0
        else:
            overfit_count += 1
        if float(val_metrics["macro_f1"]) > float(best_val_macro_f1):
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(config.patience):
                stop_reason = f"early_stopping_macro_f1_patience_{int(config.patience)}"
                break
        if int(config.overfit_patience) > 0 and overfit_count >= int(config.overfit_patience):
            stop_reason = f"early_stopping_val_loss_patience_{int(config.overfit_patience)}"
            break

    if best_state is None:
        raise RuntimeError("El entrenamiento no produjo ningún checkpoint válido.")
    model.load_state_dict(best_state)

    train_metrics = _classification_metrics(*_epoch_labels(_run_epoch(model, train_loader, criterion, device, None)), dataset.label_names)
    val_metrics = _classification_metrics(*_epoch_labels(_run_epoch(model, val_loader, criterion, device, None)), dataset.label_names)
    test_metrics = _classification_metrics(*_epoch_labels(_run_epoch(model, test_loader, criterion, device, None)), dataset.label_names)

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
        "stop_reason": stop_reason,
        "best_val_macro_f1_during_training": float(best_val_macro_f1),
        "best_val_loss_during_training": float(best_val_loss),
        "num_classes_trained": int(len(dataset.label_names)),
        "trained_labels": list(dataset.label_names),
        "missing_known_labels": [label for label in ROLE_LABELS_V1 if label not in set(dataset.label_names)],
        "used_cached_common_dataset": bool(prefer_cached_common_dataset),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    with artifacts.metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    torch.save(
        {
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
            "heuristics": {"goalkeeper_role": "POR"},
            "used_cached_common_dataset": bool(prefer_cached_common_dataset),
        },
        artifacts.checkpoint_path,
    )

    return {
        "artifacts": artifacts,
        "metrics": metrics_payload,
        "history_df": pd.DataFrame(history_rows),
        "split": split_payload,
        "checkpoint_path": artifacts.checkpoint_path,
    }


def _epoch_labels(epoch_result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return epoch_result["y_true"], epoch_result["y_pred"]


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
    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "y_true": np.concatenate(targets, axis=0),
        "y_pred": np.concatenate(preds, axis=0),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, Any]:
    labels_idx = list(range(len(label_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels_idx,
            target_names=list(label_names),
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels_idx).tolist(),
    }


def _artifact_paths(project_root: Path, output_dir: Path | None = None) -> ArtifactPaths:
    base_dir = project_root / Path("models/positions/set_transformer") if output_dir is None else Path(output_dir)
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(
        output_dir=run_dir,
        checkpoint_path=run_dir / "set_transformer_checkpoint.pt",
        history_csv_path=run_dir / "training_history.csv",
        metrics_json_path=run_dir / "metrics.json",
        split_json_path=run_dir / "split.json",
    )
