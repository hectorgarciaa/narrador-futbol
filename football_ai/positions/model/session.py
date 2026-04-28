from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn

from ..data import FeatureSpec, add_velocity_features, build_role_samples, find_project_root, infer_attack_direction_by_team
from .architecture import RoleSetTransformer
from .config import TrainingConfig
from .constraints import constrain_player_predictions_with_expected_roles
from .data_utils import FeatureStandardizer, safe_float_array
from .inference import aggregate_player_predictions, load_checkpoint, predict_probabilities
from .training import select_device


class OnlineRoleInferenceSession:
    def __init__(
        self,
        *,
        model: nn.Module,
        config: TrainingConfig,
        label_names: Sequence[str],
        standardizer: FeatureStandardizer,
        feature_spec: FeatureSpec,
        device: torch.device,
        project_root: Path,
    ) -> None:
        self.model = model
        self.config = config
        self.label_names = tuple(str(label) for label in label_names)
        self.standardizer = standardizer
        self.feature_spec = feature_spec
        self.device = device
        self.project_root = Path(project_root)

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        project_root: Path | None = None,
    ) -> "OnlineRoleInferenceSession":
        project_root = find_project_root(project_root)
        device = select_device()
        checkpoint = load_checkpoint(model_path=Path(model_path), device=device)
        config = TrainingConfig(**checkpoint["training_config"])
        label_names = tuple(str(label) for label in checkpoint["label_names"])
        feature_spec = FeatureSpec(
            tuple(str(name) for name in checkpoint["objective_feature_names"]),
            tuple(str(name) for name in checkpoint["teammate_feature_names"]),
        )
        model = RoleSetTransformer(
            objective_dim=int(len(feature_spec.objective_feature_names)),
            teammate_dim=int(len(feature_spec.teammate_feature_names)),
            num_classes=int(len(label_names)),
            config=config,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            config=config,
            label_names=label_names,
            standardizer=FeatureStandardizer.from_state(checkpoint["standardizer"]),
            feature_spec=feature_spec,
            device=device,
            project_root=project_root,
        )

    def predict_frame(
        self,
        observations: pd.DataFrame,
        expected_roles_by_team: Mapping[str, Sequence[str]] | None = None,
        batch_size: int | None = None,
        include_all_targets: bool = False,
    ) -> dict[str, Any]:
        if observations.empty:
            return {
                "frame_predictions_df": pd.DataFrame(),
                "player_predictions_df": pd.DataFrame(),
                "attack_direction_by_team": {},
                "attack_direction_details": [],
            }

        obs = observations.copy()
        if "role_label" not in obs.columns:
            obs["role_label"] = pd.NA
        if "vx" not in obs.columns or "vy" not in obs.columns:
            obs = add_velocity_features(obs)

        attack_direction_by_team, attack_details = infer_attack_direction_by_team(obs)
        samples_df, teammates_tensor, teammate_mask, feature_spec = build_role_samples(
            observations_with_roles=obs,
            attack_direction_by_team=attack_direction_by_team,
            max_teammates=int(self.config.max_teammates),
            drop_unlabeled=False,
            include_all_targets=bool(include_all_targets),
        )
        if samples_df.empty:
            return {
                "frame_predictions_df": pd.DataFrame(),
                "player_predictions_df": pd.DataFrame(),
                "attack_direction_by_team": {str(team): int(direction) for team, direction in attack_direction_by_team.items()},
                "attack_direction_details": attack_details.to_dict(orient="records"),
            }
        if feature_spec != self.feature_spec:
            raise ValueError("FeatureSpec de inferencia online no coincide con el checkpoint.")

        sample_meta = obs.loc[:, [col for col in (
            "match_id", "frame_id", "team_id", "player_id", "class_name", "x_m", "y_m",
            "confidence_tracking", "visible", "is_interpolated", "role_inference_target",
        ) if col in obs.columns]].drop_duplicates(
            subset=["match_id", "frame_id", "team_id", "player_id"],
            keep="last",
        )
        samples_df = samples_df.merge(sample_meta, on=["match_id", "frame_id", "team_id", "player_id"], how="left")
        objective = safe_float_array(samples_df.loc[:, list(self.feature_spec.objective_feature_names)].to_numpy(dtype="float32"))
        objective_scaled, teammates_scaled = self.standardizer.transform(
            objective=objective,
            teammates=teammates_tensor,
            teammate_mask=teammate_mask,
        )

        probs = predict_probabilities(
            model=self.model,
            objective=objective_scaled,
            teammates=teammates_scaled,
            teammate_mask=teammate_mask,
            batch_size=int(batch_size or self.config.batch_size),
            device=self.device,
        )
        frame_predictions_df = samples_df.copy()
        for idx, label in enumerate(self.label_names):
            frame_predictions_df[f"prob_{label}"] = probs[:, idx]
        frame_predictions_df["predicted_role_frame"] = [self.label_names[int(idx)] for idx in probs.argmax(axis=1)]
        frame_predictions_df["predicted_role_frame_confidence"] = probs.max(axis=1).astype("float32")
        if "class_name" in frame_predictions_df.columns:
            goalkeeper_mask = frame_predictions_df["class_name"].astype(str) == "goalkeeper"
            frame_predictions_df.loc[goalkeeper_mask, "predicted_role_frame"] = "POR"
            frame_predictions_df.loc[goalkeeper_mask, "predicted_role_frame_confidence"] = 1.0

        player_predictions_df = aggregate_player_predictions(frame_predictions_df, self.label_names)
        if expected_roles_by_team:
            player_predictions_df = constrain_player_predictions_with_expected_roles(
                player_predictions_df=player_predictions_df,
                label_names=self.label_names,
                expected_roles_by_team=expected_roles_by_team,
            )
        merge_cols = [
            "team_id",
            "player_id",
            "predicted_role",
            "predicted_role_confidence",
            "predicted_role_unconstrained",
            "predicted_role_confidence_unconstrained",
            "matched_model_role",
            "expected_role_slot",
            "assignment_method",
            "assignment_cost",
        ]
        available_merge_cols = [col for col in merge_cols if col in player_predictions_df.columns]
        if available_merge_cols:
            frame_predictions_df = frame_predictions_df.merge(
                player_predictions_df[available_merge_cols],
                on=["team_id", "player_id"],
                how="left",
            )
        return {
            "frame_predictions_df": frame_predictions_df,
            "player_predictions_df": player_predictions_df,
            "attack_direction_by_team": {str(team): int(direction) for team, direction in attack_direction_by_team.items()},
            "attack_direction_details": attack_details.to_dict(orient="records"),
        }
