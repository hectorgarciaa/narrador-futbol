from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from football_ai.core.config import Config

from ..data import (
    add_velocity_features,
    build_observations_from_tracks,
    find_project_root,
    load_tracks_json,
    resolve_tracks_path_for_video,
)
from .architecture import RoleSetTransformer
from .constraints import constrain_player_predictions_with_expected_roles
from .data_utils import FeatureStandardizer, RoleDataset, safe_float_array
from .config import DEFAULT_MAX_TEAMMATES, DEFAULT_PREDICTIONS_DIR, DatasetBundle, TrainingConfig
from .training import select_device


def load_checkpoint(model_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(model_path), map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint inválido: {model_path}")
    return dict(checkpoint)


def predict_probabilities(
    model: nn.Module,
    objective: np.ndarray,
    teammates: np.ndarray,
    teammate_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        RoleDataset(objective=objective, teammates=teammates, teammate_mask=teammate_mask, labels=None),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )
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


def aggregate_player_predictions(frame_df: pd.DataFrame, label_names: Sequence[str]) -> pd.DataFrame:
    prob_cols = [f"prob_{label}" for label in label_names]
    rows: list[dict[str, Any]] = []
    for (team_id, player_id), group in frame_df.groupby(["team_id", "player_id"], sort=True):
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
            for coord_col in ("x", "y", "x_m", "y_m"):
                if coord_col in group.columns:
                    row[coord_col] = float(pd.to_numeric(group[coord_col], errors="coerce").mean())
            row.update({prob_col: 0.0 for prob_col in prob_cols})
            rows.append(row)
            continue
        mean_probs = group[prob_cols].mean(axis=0)
        row = {
            "team_id": str(team_id),
            "player_id": int(player_id),
            "class_name": class_name,
            "predicted_role": str(mean_probs.idxmax()).replace("prob_", "", 1),
            "predicted_role_confidence": float(mean_probs.max()),
            "frames_seen": int(len(group)),
        }
        for coord_col in ("x", "y", "x_m", "y_m"):
            if coord_col in group.columns:
                row[coord_col] = float(pd.to_numeric(group[coord_col], errors="coerce").mean())
        row.update({prob_col: float(mean_probs[prob_col]) for prob_col in prob_cols})
        rows.append(row)
    return pd.DataFrame(rows)


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
    if expected_roles_by_team is None:
        expected_roles_by_team = expected_roles_by_team_from_config(project_root)
    dataset, attack_details, attack_direction_by_team, resolved_tracks_path, checkpoint, device, label_names = _prepare_inference_run(
        model_path=model_path,
        project_root=project_root,
        video_path=video_path,
        tracks_path=tracks_path,
    )
    frame_predictions_df, player_predictions_df = _predict_dataframes(
        checkpoint=checkpoint,
        dataset=dataset,
        device=device,
        label_names=label_names,
        batch_size=batch_size,
        expected_roles_by_team=expected_roles_by_team,
    )
    augmented_tracks = augment_tracks_with_predictions(
        tracks=load_tracks_json(resolved_tracks_path),
        frame_predictions_df=frame_predictions_df,
        player_predictions_df=player_predictions_df,
    )
    out_dir = ((project_root / DEFAULT_PREDICTIONS_DIR) if output_dir is None else Path(output_dir)) / f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_csv_path = out_dir / "frame_role_predictions.csv"
    player_csv_path = out_dir / "player_role_summary.csv"
    tracks_json_path = out_dir / "tracks_with_predicted_roles.json"
    meta_json_path = out_dir / "inference_meta.json"
    frame_predictions_df.to_csv(frame_csv_path, index=False)
    player_predictions_df.to_csv(player_csv_path, index=False)
    with tracks_json_path.open("w", encoding="utf-8") as f:
        json.dump(augmented_tracks, f, ensure_ascii=False, indent=2)
    _write_meta(
        meta_json_path=meta_json_path,
        model_path=model_path,
        video_path=video_path,
        tracks_path=resolved_tracks_path,
        output_dir=out_dir,
        device=device,
        frame_predictions_df=frame_predictions_df,
        player_predictions_df=player_predictions_df,
        label_names=label_names,
        checkpoint=checkpoint,
        expected_roles_by_team=expected_roles_by_team,
        attack_direction_by_team=attack_direction_by_team,
        attack_details=attack_details,
    )
    return {
        "output_dir": out_dir,
        "frame_predictions_path": frame_csv_path,
        "player_predictions_path": player_csv_path,
        "tracks_with_roles_path": tracks_json_path,
        "meta_path": meta_json_path,
        "frame_predictions_df": frame_predictions_df,
        "player_predictions_df": player_predictions_df,
    }


def predict_roles_for_tracks_payload(
    model_path: Path,
    tracks: Mapping[str, Any],
    video_path: Path,
    project_root: Path | None = None,
    batch_size: int = 1024,
    expected_roles_by_team: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    video_path = Path(video_path)
    if expected_roles_by_team is None:
        expected_roles_by_team = expected_roles_by_team_from_config(project_root)
    device = select_device()
    checkpoint = load_checkpoint(model_path=model_path, device=device)
    config = TrainingConfig(**checkpoint["training_config"])
    label_names = tuple(str(label) for label in checkpoint["label_names"])
    dataset, attack_details, attack_direction_by_team = prepare_inference_dataset_from_tracks_payload(
        tracks=tracks,
        video_path=video_path,
        max_teammates=int(config.max_teammates),
    )
    frame_predictions_df, player_predictions_df = _predict_dataframes(
        checkpoint=checkpoint,
        dataset=dataset,
        device=device,
        label_names=label_names,
        batch_size=batch_size,
        expected_roles_by_team=expected_roles_by_team,
    )
    return {
        "device": str(device),
        "frame_predictions_df": frame_predictions_df,
        "player_predictions_df": player_predictions_df,
        "tracks_with_roles": augment_tracks_with_predictions(tracks, frame_predictions_df, player_predictions_df),
        "attack_direction_by_team": {str(team): int(direction) for team, direction in attack_direction_by_team.items()},
        "attack_direction_details": attack_details.to_dict(orient="records"),
        "trained_labels": list(label_names),
        "goalkeeper_heuristic_label": checkpoint.get("heuristics", {}).get("goalkeeper_role", "POR"),
        "num_frame_predictions": int(len(frame_predictions_df)),
        "num_player_predictions": int(len(player_predictions_df)),
        "expected_roles_by_team": (
            {str(team_id): [str(role) for role in roles] for team_id, roles in expected_roles_by_team.items()}
            if expected_roles_by_team else None
        ),
        "stable_assignment_method": "hungarian_expected_roles" if expected_roles_by_team else "unconstrained_softmax",
        "model_path": str(model_path),
        "video_path": str(video_path),
    }


def prepare_inference_dataset(
    project_root: Path,
    video_path: Path,
    tracks_path: Path | None = None,
    max_teammates: int = DEFAULT_MAX_TEAMMATES,
) -> tuple[DatasetBundle, pd.DataFrame, dict[str, int], Path]:
    resolved_tracks_path = tracks_path or resolve_tracks_path_for_video(project_root, video_path)
    if not resolved_tracks_path.exists():
        raise FileNotFoundError(f"No existe tracks JSON para inferencia: {resolved_tracks_path}")
    return (*prepare_inference_dataset_from_tracks_payload(load_tracks_json(resolved_tracks_path), video_path, max_teammates), resolved_tracks_path)


def prepare_inference_dataset_from_tracks_payload(
    tracks: Mapping[str, Any],
    video_path: Path,
    max_teammates: int = DEFAULT_MAX_TEAMMATES,
) -> tuple[DatasetBundle, pd.DataFrame, dict[str, int]]:
    observations = add_velocity_features(
        build_observations_from_tracks(
            tracks=tracks,
            match_id=video_path.stem.replace(" ", "_"),
            tracked_classes=("player", "goalkeeper"),
        )
    )
    observations["role_label"] = pd.NA
    if observations.empty:
        raise ValueError("No hay observaciones con field_position_m para inferencia.")
    from ..data import build_role_samples, infer_attack_direction_by_team
    attack_direction_by_team, attack_details = infer_attack_direction_by_team(observations)
    samples_df, teammates_tensor, teammate_mask, feature_spec = build_role_samples(
        observations_with_roles=observations,
        attack_direction_by_team=attack_direction_by_team,
        max_teammates=int(max_teammates),
        drop_unlabeled=False,
    )
    if samples_df.empty:
        raise ValueError("No se pudieron construir samples de inferencia.")
    sample_meta = observations[["match_id", "frame_id", "team_id", "player_id", "class_name", "x_m", "y_m", "confidence_tracking"]].drop_duplicates(
        subset=["match_id", "frame_id", "team_id", "player_id"],
        keep="last",
    )
    samples_df = samples_df.merge(sample_meta, on=["match_id", "frame_id", "team_id", "player_id"], how="left")
    return (
        DatasetBundle(
            samples_df=samples_df.reset_index(drop=True),
            objective=safe_float_array(samples_df.loc[:, list(feature_spec.objective_feature_names)].to_numpy(dtype=np.float32)),
            teammates=np.asarray(teammates_tensor, dtype=np.float32),
            teammate_mask=np.asarray(teammate_mask, dtype=bool),
            feature_spec=feature_spec,
            label_names=tuple(),
        ),
        attack_details,
        attack_direction_by_team,
    )


def augment_tracks_with_predictions(
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
        payload = {
            "predicted_role": str(row.predicted_role),
            "predicted_role_confidence": float(row.predicted_role_confidence),
        }
        for attr in (
            "predicted_role_unconstrained",
            "predicted_role_confidence_unconstrained",
            "matched_model_role",
            "expected_role_slot",
            "assignment_method",
        ):
            if hasattr(row, attr):
                value = getattr(row, attr)
                if pd.notna(value):
                    payload[attr] = float(value) if attr.endswith("confidence_unconstrained") else str(value)
        player_map[(str(row.team_id), int(row.player_id))] = payload
    for class_name in ("player", "goalkeeper"):
        frames = output.get(class_name, [])
        if not isinstance(frames, list):
            continue
        for frame_id, frame_tracks in enumerate(frames):
            if not isinstance(frame_tracks, Mapping):
                continue
            for player_id_raw, track_data in list(frame_tracks.items()):
                if not isinstance(track_data, Mapping):
                    continue
                try:
                    player_id = int(player_id_raw)
                except (TypeError, ValueError):
                    continue
                team_id = track_data.get("team")
                if team_id is None:
                    continue
                enriched = dict(track_data)
                stable = player_map.get((str(team_id), player_id))
                frame_pred = frame_map.get((int(frame_id), str(team_id), player_id))
                if stable is not None:
                    enriched.update(stable)
                if frame_pred is not None:
                    enriched["predicted_role_frame"] = frame_pred[0]
                    enriched["predicted_role_frame_confidence"] = frame_pred[1]
                frame_tracks[player_id_raw] = enriched
    return output


def expected_roles_by_team_from_config(project_root: Path) -> dict[str, list[str]] | None:
    raw_mapping = Config.from_yaml(project_root / "config.yaml").get("positions", "expected_roles_by_team", default=None)
    if not isinstance(raw_mapping, dict):
        return None
    normalized = {}
    for team_id, roles in raw_mapping.items():
        if isinstance(roles, (list, tuple)):
            normalized[str(team_id)] = [str(role) for role in roles]
    return normalized or None


def _prepare_inference_run(
    model_path: Path,
    project_root: Path,
    video_path: Path,
    tracks_path: Path | None,
) -> tuple[DatasetBundle, pd.DataFrame, dict[str, int], Path, dict[str, Any], torch.device, tuple[str, ...]]:
    device = select_device()
    checkpoint = load_checkpoint(model_path=model_path, device=device)
    config = TrainingConfig(**checkpoint["training_config"])
    label_names = tuple(str(label) for label in checkpoint["label_names"])
    dataset, attack_details, attack_direction_by_team, resolved_tracks_path = prepare_inference_dataset(
        project_root=project_root,
        video_path=video_path,
        tracks_path=tracks_path,
        max_teammates=int(config.max_teammates),
    )
    _validate_feature_spec(dataset, checkpoint)
    return dataset, attack_details, attack_direction_by_team, resolved_tracks_path, checkpoint, device, label_names


def _predict_dataframes(
    checkpoint: dict[str, Any],
    dataset: DatasetBundle,
    device: torch.device,
    label_names: tuple[str, ...],
    batch_size: int,
    expected_roles_by_team: Mapping[str, Sequence[str]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standardizer = FeatureStandardizer.from_state(checkpoint["standardizer"])
    objective_scaled, teammates_scaled = standardizer.transform(
        objective=dataset.objective,
        teammates=dataset.teammates,
        teammate_mask=dataset.teammate_mask,
    )
    model = RoleSetTransformer(
        objective_dim=int(objective_scaled.shape[1]),
        teammate_dim=int(teammates_scaled.shape[2]),
        num_classes=int(len(label_names)),
        config=TrainingConfig(**checkpoint["training_config"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    probs = predict_probabilities(model, objective_scaled, teammates_scaled, dataset.teammate_mask, int(batch_size), device)

    frame_predictions_df = dataset.samples_df.copy()
    for idx, label in enumerate(label_names):
        frame_predictions_df[f"prob_{label}"] = probs[:, idx]
    frame_predictions_df["predicted_role_frame"] = [label_names[int(idx)] for idx in probs.argmax(axis=1)]
    frame_predictions_df["predicted_role_frame_confidence"] = probs.max(axis=1).astype(np.float32)
    goalkeeper_mask = frame_predictions_df["class_name"].astype(str) == "goalkeeper"
    frame_predictions_df.loc[goalkeeper_mask, "predicted_role_frame"] = "POR"
    frame_predictions_df.loc[goalkeeper_mask, "predicted_role_frame_confidence"] = 1.0

    player_predictions_df = constrain_player_predictions_with_expected_roles(
        player_predictions_df=aggregate_player_predictions(frame_predictions_df, label_names),
        label_names=label_names,
        expected_roles_by_team=expected_roles_by_team,
    )
    merge_cols = [
        "team_id", "player_id", "predicted_role", "predicted_role_confidence",
        "predicted_role_unconstrained", "predicted_role_confidence_unconstrained",
        "matched_model_role", "expected_role_slot", "assignment_method", "assignment_cost",
    ]
    frame_predictions_df = frame_predictions_df.merge(player_predictions_df[merge_cols], on=["team_id", "player_id"], how="left")
    return frame_predictions_df, player_predictions_df


def _validate_feature_spec(dataset: DatasetBundle, checkpoint: dict[str, Any]) -> None:
    if list(dataset.feature_spec.objective_feature_names) != list(checkpoint["objective_feature_names"]):
        raise ValueError("Las objective_feature_names de inferencia no coinciden con el checkpoint.")
    if list(dataset.feature_spec.teammate_feature_names) != list(checkpoint["teammate_feature_names"]):
        raise ValueError("Las teammate_feature_names de inferencia no coinciden con el checkpoint.")


def _write_meta(
    meta_json_path: Path,
    model_path: Path,
    video_path: Path,
    tracks_path: Path,
    output_dir: Path,
    device: torch.device,
    frame_predictions_df: pd.DataFrame,
    player_predictions_df: pd.DataFrame,
    label_names: tuple[str, ...],
    checkpoint: dict[str, Any],
    expected_roles_by_team: Mapping[str, Sequence[str]] | None,
    attack_direction_by_team: Mapping[str, int],
    attack_details: pd.DataFrame,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(),
        "model_path": str(model_path),
        "video_path": str(video_path),
        "tracks_path": str(tracks_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "num_frame_predictions": int(len(frame_predictions_df)),
        "num_player_predictions": int(len(player_predictions_df)),
        "trained_labels": list(label_names),
        "goalkeeper_heuristic_label": checkpoint.get("heuristics", {}).get("goalkeeper_role", "POR"),
        "expected_roles_by_team": (
            {str(team_id): [str(role) for role in roles] for team_id, roles in expected_roles_by_team.items()}
            if expected_roles_by_team else None
        ),
        "stable_assignment_method": "hungarian_expected_roles" if expected_roles_by_team else "unconstrained_softmax",
        "attack_direction_by_team": {str(team): int(direction) for team, direction in attack_direction_by_team.items()},
        "attack_direction_details": attack_details.to_dict(orient="records"),
    }
    with meta_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
