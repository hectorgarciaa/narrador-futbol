from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from ..data import (
    ROLE_LABELS_V1,
    FeatureSpec,
    add_velocity_features,
    build_role_samples,
    find_project_root,
    infer_attack_direction_by_team,
)
from .config import DEFAULT_BASE_TABLE_PATH, DatasetBundle, SplitBundle, TrainingConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_float_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def ordered_label_names(labels: Iterable[str]) -> tuple[str, ...]:
    labels_set = {str(label) for label in labels}
    ordered = [label for label in ROLE_LABELS_V1 if label in labels_set]
    return tuple(ordered + sorted(labels_set.difference(ordered)))


def load_labeled_dataset_from_base_table(
    project_root: Path | None = None,
    base_table_path: Path | None = None,
    max_teammates: int = 10,
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
    if "team_id" not in base_df.columns or "match_id" not in base_df.columns:
        raise ValueError(f"{path} debe contener 'match_id' y 'team_id'.")

    base_df["role_label"] = base_df["role_label"].astype("string").str.strip()
    base_df.loc[base_df["role_label"] == "", "role_label"] = pd.NA
    base_df = base_df[base_df["role_label"].notna()].copy()
    if base_df.empty:
        raise ValueError(f"{path} no contiene filas etiquetadas.")

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
    objective = safe_float_array(samples_df.loc[:, list(feature_spec.objective_feature_names)].to_numpy(dtype=np.float32))
    return DatasetBundle(
        samples_df=samples_df.reset_index(drop=True),
        objective=objective,
        teammates=teammates,
        teammate_mask=teammate_mask,
        feature_spec=feature_spec,
        label_names=ordered_label_names(samples_df["label"].astype(str).tolist()),
    )


def _load_cached_common_dataset_if_available(path: Path, max_teammates: int) -> DatasetBundle | None:
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
        raise ValueError("El cache común de samples está desalineado con el CSV de metadata.")

    return DatasetBundle(
        samples_df=samples_df.reset_index(drop=True),
        objective=safe_float_array(samples_df.loc[:, list(objective_feature_names)].to_numpy(dtype=np.float32)),
        teammates=teammates,
        teammate_mask=teammate_mask,
        feature_spec=FeatureSpec(
            objective_feature_names=objective_feature_names,
            teammate_feature_names=teammate_feature_names,
        ),
        label_names=ordered_label_names(samples_df["label"].astype(str).tolist()),
    )


def split_dataset_by_match(dataset: DatasetBundle, seed: int = 42) -> SplitBundle:
    groups = dataset.samples_df["match_id"].astype(str).to_numpy()
    if np.unique(groups).size < 3:
        raise ValueError("Se necesitan al menos 3 partidos/clips distintos para train/val/test por grupos.")

    all_labels = set(dataset.samples_df["label"].astype(str).tolist())
    indices = np.arange(len(dataset.samples_df))
    for attempt in range(24):
        current_seed = int(seed + attempt)
        train_val_idx, test_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=1, random_state=current_seed).split(indices, groups=groups)
        )
        train_idx, val_idx = _split_train_val(train_val_idx, groups[train_val_idx], current_seed + 97)
        if set(dataset.samples_df.iloc[train_idx]["label"].astype(str).tolist()) != all_labels:
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


def _split_train_val(indices: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_rel_idx, val_rel_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=1, random_state=seed).split(indices, groups=groups)
    )
    return indices[train_rel_idx], indices[val_rel_idx]


class FeatureStandardizer:
    def __init__(
        self,
        objective_mean: np.ndarray,
        objective_std: np.ndarray,
        teammate_mean: np.ndarray,
        teammate_std: np.ndarray,
    ) -> None:
        self.objective_mean = safe_float_array(objective_mean).reshape(1, -1)
        self.objective_std = safe_float_array(objective_std).reshape(1, -1)
        self.teammate_mean = safe_float_array(teammate_mean).reshape(1, 1, -1)
        self.teammate_std = safe_float_array(teammate_std).reshape(1, 1, -1)

    @classmethod
    def fit(cls, objective: np.ndarray, teammates: np.ndarray, teammate_mask: np.ndarray) -> "FeatureStandardizer":
        objective = safe_float_array(objective)
        teammates = safe_float_array(teammates)
        teammate_mask = np.asarray(teammate_mask, dtype=bool)

        objective_mean = objective.mean(axis=0)
        objective_std = np.where(objective.std(axis=0) < 1e-6, 1.0, objective.std(axis=0))
        if teammate_mask.any():
            teammate_values = teammates[teammate_mask]
            teammate_mean = teammate_values.mean(axis=0)
            teammate_std = np.where(teammate_values.std(axis=0) < 1e-6, 1.0, teammate_values.std(axis=0))
        else:
            teammate_dim = teammates.shape[-1]
            teammate_mean = np.zeros((teammate_dim,), dtype=np.float32)
            teammate_std = np.ones((teammate_dim,), dtype=np.float32)
        return cls(objective_mean, objective_std, teammate_mean, teammate_std)

    def transform(
        self,
        objective: np.ndarray,
        teammates: np.ndarray,
        teammate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        objective_scaled = (safe_float_array(objective) - self.objective_mean) / self.objective_std
        teammates_scaled = (safe_float_array(teammates) - self.teammate_mean) / self.teammate_std
        teammates_scaled = np.where(np.asarray(teammate_mask, dtype=bool)[..., None], teammates_scaled, 0.0)
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


def build_dataloaders(
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
    train_idx = _subset_indices_keep_labels(split.train_idx, max_train_samples, config.seed, labels_idx)
    val_idx = _subset_indices(split.val_idx, max_val_samples, config.seed + 1)
    test_idx = _subset_indices(split.test_idx, max_test_samples, config.seed + 2)
    datasets = {
        "train": RoleDataset(objective_scaled[train_idx], teammates_scaled[train_idx], teammate_mask[train_idx], labels_idx[train_idx]),
        "val": RoleDataset(objective_scaled[val_idx], teammates_scaled[val_idx], teammate_mask[val_idx], labels_idx[val_idx]),
        "test": RoleDataset(objective_scaled[test_idx], teammates_scaled[test_idx], teammate_mask[test_idx], labels_idx[test_idx]),
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


def _subset_indices(indices: np.ndarray, max_size: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_size is None or max_size <= 0 or len(indices) <= int(max_size):
        return indices
    return np.sort(np.random.default_rng(int(seed)).choice(indices, size=int(max_size), replace=False)).astype(np.int64)


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
        raise ValueError(f"max_size={max_size} es menor que el número de clases presentes ({len(present_labels)}).")

    rng = np.random.default_rng(int(seed))
    mandatory = np.asarray(
        [int(rng.choice(indices[labels_idx[indices] == int(label)], size=1, replace=False)[0]) for label in present_labels.tolist()],
        dtype=np.int64,
    )
    extra_needed = int(max_size) - len(mandatory)
    if extra_needed <= 0:
        return np.sort(mandatory)

    remaining_pool = np.array(sorted(set(indices.tolist()).difference(mandatory.tolist())), dtype=np.int64)
    if len(remaining_pool) == 0:
        return np.sort(mandatory)
    extra = rng.choice(remaining_pool, size=min(extra_needed, len(remaining_pool)), replace=False)
    return np.sort(np.concatenate([mandatory, extra])).astype(np.int64)
