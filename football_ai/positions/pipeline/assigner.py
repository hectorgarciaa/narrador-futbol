"""
Motor online de roles posicionales basado en doble pasada de Húngaro.
"""

from __future__ import annotations

import itertools
import math
import re
import shutil
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover - fallback para entornos mínimos
    _scipy_linear_sum_assignment = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - entorno sin matplotlib
    matplotlib = None
    plt = None

from .lineup_spec import LineupSlotMatcher, base_role_token, normalize_slot_token


DEFAULT_SPECIAL_SEED_CANONICAL_IDS = (1, 2)
DEFAULT_SPECIAL_SEED_DEFENDER_ROLES = (
    "CD",
    "CI",
    "LD",
    "LI",
    "DFC_DER",
    "DFC_IZQ",
    "DFC_CENT",
)
DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH = (
    "models/positions/set_transformer/20260317_211507/set_transformer_checkpoint.pt"
)
DEFAULT_EXPECTED_ROLES_BY_TEAM = {
    "Real Madrid": [
        "POR",
        "LD",
        "LI",
        "DFC_DER",
        "DFC_IZQ",
        "DFC_CENT",
        "MC",
        "MI",
        "MD",
        "DC",
        "DC",
    ],
    "Wolfsburgo": [
        "POR",
        "LD",
        "LI",
        "DFC_DER",
        "DFC_IZQ",
        "DFC_CENT",
        "MC",
        "MI",
        "MD",
        "DC",
        "DC",
    ],
}
SEGMENT_POSITION_WEIGHT = 1.25
SEGMENT_CONFIDENCE_WEIGHT = 0.35


def sanitize_video_stem(raw_stem):
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def build_role_artifacts_output_dir(config, video_path):
    role_artifacts_root = config.get_path(
        "paths", "output", "role_artifacts", create_if_missing=True
    )
    role_artifacts_root.mkdir(parents=True, exist_ok=True)
    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    role_artifacts_dir = role_artifacts_root / f"{sanitized_stem}_role_artifacts"
    role_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return role_artifacts_dir


def build_role_predictions_output_paths(config, video_path, use_artifacts_dir=True):
    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    if use_artifacts_dir:
        output_dir = build_role_artifacts_output_dir(config, video_path)
    else:
        output_dir = (
            config.get_path("paths", "output", "tracks_json", create_if_missing=True)
            / "tracker"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    frame_csv_path = output_dir / f"{sanitized_stem}_frame_role_predictions.csv"
    player_csv_path = output_dir / f"{sanitized_stem}_player_role_summary.csv"
    greedy_csv_path = output_dir / f"{sanitized_stem}_greedy_role_diagnostics.csv"
    return str(frame_csv_path), str(player_csv_path), str(greedy_csv_path)


def save_dataframe_csv(df, output_path, logger, description):
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("%s saved to: %s", description, output_path)
    except Exception as exc:  # pragma: no cover - logging defensivo
        logger.error("Error saving %s to %s: %s", description, output_path, exc)


def copy_output_artifact(source_path, target_path, logger, description):
    try:
        source_path = Path(source_path)
        target_path = Path(target_path)
        if source_path.resolve() == target_path.resolve():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        logger.info("%s copied to: %s", description, target_path)
    except Exception as exc:  # pragma: no cover - logging defensivo
        logger.error(
            "Error copying %s from %s to %s: %s",
            description,
            source_path,
            target_path,
            exc,
        )


def _normalize_expected_roles_mapping(expected_roles_by_team):
    if not expected_roles_by_team:
        return {}
    normalized = {}
    for team_name, roles in dict(expected_roles_by_team).items():
        normalized[str(team_name)] = [
            normalize_slot_token(role)
            for role in list(roles or [])
            if normalize_slot_token(role)
        ]
    return normalized


def _resolve_expected_roles_by_team_from_config(config):
    tracking_cfg = getattr(config, "tracking", {}) or {}
    configured = tracking_cfg.get("expected_roles_by_team")
    if configured:
        return _normalize_expected_roles_mapping(configured)
    return _normalize_expected_roles_mapping(DEFAULT_EXPECTED_ROLES_BY_TEAM)


def _safe_field_position_m(track_data):
    if not isinstance(track_data, dict):
        return None
    field_position = track_data.get("field_position_m")
    if field_position is None:
        return None
    arr = np.asarray(field_position, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return float(arr[0]), float(arr[1])


def _frame_track_keys(frame_tracks, track_id):
    return (track_id, str(track_id))


def _special_seed_frame_track_keys(frame_tracks, special_id):
    keys = []
    for track_key in _frame_track_keys(frame_tracks, special_id):
        if track_key in frame_tracks:
            keys.append(track_key)
    return keys


def _first_existing_track_payload(tracks_frame, track_id):
    for class_name in ("player", "goalkeeper"):
        frame_tracks = tracks_frame.get(class_name, {})
        if not isinstance(frame_tracks, dict):
            continue
        for track_key in _frame_track_keys(frame_tracks, track_id):
            payload = frame_tracks.get(track_key)
            if isinstance(payload, dict):
                return class_name, track_key, payload
    return None, None, None


def _segment_anchor_for_slot(layout_by_team, team_id, slot_name):
    team_layout = layout_by_team.get(str(team_id), {})
    coords = team_layout.get(normalize_slot_token(slot_name))
    if not isinstance(coords, dict):
        return None
    x_value = pd.to_numeric(coords.get("x"), errors="coerce")
    y_value = pd.to_numeric(coords.get("y"), errors="coerce")
    if pd.isna(x_value) or pd.isna(y_value):
        return None
    return float(x_value) / 100.0, float(y_value) / 100.0


def _solve_assignment(cost_matrix):
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)

    num_rows, num_cols = cost_matrix.shape
    candidate_cols = range(num_cols)
    best_cost = None
    best_pairs = None
    for chosen_cols in itertools.permutations(candidate_cols, min(num_rows, num_cols)):
        total_cost = 0.0
        pairs = []
        for row_idx, col_idx in enumerate(chosen_cols):
            total_cost += float(cost_matrix[row_idx, col_idx])
            pairs.append((row_idx, col_idx))
        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_pairs = pairs
    if not best_pairs:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    row_ind = np.asarray([pair[0] for pair in best_pairs], dtype=np.int64)
    col_ind = np.asarray([pair[1] for pair in best_pairs], dtype=np.int64)
    return row_ind, col_ind


def save_role_visualizations(
    frame_df,
    player_df,
    config,
    video_path,
    output_dir,
    expected_roles_by_team,
    assignment_method,
    min_count,
    min_cumulative_ratio,
    min_final_ratio,
    logger,
    legacy_output_dir=None,
):
    if plt is None:
        logger.info("Se omiten PNG de roles: matplotlib no está disponible.")
        return
    if player_df is None or player_df.empty:
        logger.info("Se omiten PNG de roles: no hay player role summary.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / f"{sanitize_video_stem(Path(video_path).stem)}_role_summary.png"

    plot_df = player_df.copy()
    role_col = (
        "display_role_slot"
        if "display_role_slot" in plot_df.columns
        else "predicted_role"
        if "predicted_role" in plot_df.columns
        else None
    )
    if role_col is None:
        logger.info("Se omiten PNG de roles: no hay columna de rol estable.")
        return

    counts = (
        plot_df.groupby(["team_id", role_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["team_id", "count", role_col], ascending=[True, False, True])
    )

    teams = counts["team_id"].astype(str).unique().tolist()
    fig, axes = plt.subplots(
        nrows=max(1, len(teams)),
        ncols=1,
        figsize=(12, max(4, 3 * max(1, len(teams)))),
        squeeze=False,
    )
    for axis, team_id in zip(axes.reshape(-1), teams):
        team_df = counts[counts["team_id"].astype(str) == str(team_id)]
        axis.bar(
            team_df[role_col].astype(str).tolist(),
            team_df["count"].astype(int).tolist(),
            color="#1d3557",
        )
        axis.set_title(
            f"{team_id} | segment-level={assignment_method} | min_count={min_count}"
        )
        axis.set_ylabel("segmentos")
        axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Role summary PNG saved to: %s", chart_path)

    if legacy_output_dir is not None:
        copy_output_artifact(
            chart_path,
            Path(legacy_output_dir) / chart_path.name,
            logger,
            "Role summary PNG",
        )


def apply_special_seed_role_team_assignment(tracks, config, video_path, logger):
    tracking_cfg = getattr(config, "tracking", {}) or {}
    if not tracking_cfg.get("special_seed_role_team_assignment_enabled", True):
        return tracks, None

    model_path = Path(config.project_root) / tracking_cfg.get(
        "special_seed_role_model_path",
        DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH,
    )
    if not model_path.exists():
        logger.warning(
            "Se omite special_seed_role_team_assignment: no existe el checkpoint %s",
            model_path,
        )
        return tracks, None

    try:
        from ..model.train import (
            predict_roles_for_tracks_payload,
        )
    except Exception as exc:
        logger.warning(
            "No se pudo importar el pipeline de roles posicionales: %s",
            exc,
        )
        return tracks, None

    try:
        role_result = predict_roles_for_tracks_payload(
            model_path=model_path,
            tracks=tracks,
            video_path=Path(video_path),
            project_root=config.project_root,
        )
    except Exception as exc:
        logger.warning("Falló el postproceso de roles posicionales: %s", exc)
        return tracks, None
    return role_result.get("tracks_with_roles", tracks), role_result


class OnlineSpecialSeedRoleAssigner:
    def __init__(
        self,
        config,
        video_path,
        logger,
        *,
        expected_roles_by_team_override=None,
        lineup_matcher=None,
    ):
        self.config = config
        self.video_path = Path(video_path)
        self.logger = logger
        tracking_cfg = getattr(config, "tracking", {}) or {}
        self.enabled = bool(
            tracking_cfg.get("special_seed_role_team_assignment_enabled", True)
        )
        self.special_ids = tuple(
            int(track_id)
            for track_id in tracking_cfg.get(
                "special_seed_canonical_ids",
                list(DEFAULT_SPECIAL_SEED_CANONICAL_IDS),
            )
        )
        self.defender_roles = {
            normalize_slot_token(role)
            for role in tracking_cfg.get(
                "special_seed_defender_roles",
                list(DEFAULT_SPECIAL_SEED_DEFENDER_ROLES),
            )
        }
        self.expected_roles_by_team = (
            _normalize_expected_roles_mapping(expected_roles_by_team_override)
            if expected_roles_by_team_override is not None
            else _resolve_expected_roles_by_team_from_config(config)
        )
        self.lineup_matcher = (
            lineup_matcher
            if isinstance(lineup_matcher, LineupSlotMatcher)
            else None
        )
        self.segment_expected_slots_by_team = self._build_segment_expected_slots_by_team()
        self.pitch_layout_by_team = self._build_pitch_layout_by_team()
        self.segment_switch_distance_m = float(
            tracking_cfg.get(
                "role_segment_switch_distance_m",
                tracking_cfg.get("role_swap_position_jump_m", 14.0),
            )
        )
        self.segment_min_observations = max(
            2,
            int(
                tracking_cfg.get(
                    "role_segment_min_observations",
                    tracking_cfg.get("role_swap_min_recent_samples", 6),
                )
            ),
        )
        self.recent_window = max(
            3,
            int(tracking_cfg.get("role_segment_recent_window", 10)),
        )
        self.role_stabilization_expected_roles_assignment = "double_hungarian_segments"
        self.role_stabilization_expected_roles_min_count = 1
        self.role_stabilization_expected_roles_min_ratio = 0.0
        self.role_stabilization_expected_roles_min_final_ratio = 0.0
        self._reset_runtime_state()
        self.role_session = None

        if not self.enabled:
            return

        model_path = Path(config.project_root) / tracking_cfg.get(
            "special_seed_role_model_path",
            DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH,
        )
        if not model_path.exists():
            self.logger.warning(
                "Se desactiva online special-seed role assignment: no existe el checkpoint %s",
                model_path,
            )
            self.enabled = False
            return

        try:
            from ..model.train import (
                OnlineRoleInferenceSession,
            )
        except Exception as exc:
            self.logger.warning(
                "Se desactiva online special-seed role assignment: no se pudo importar OnlineRoleInferenceSession (%s)",
                exc,
            )
            self.enabled = False
            return

        try:
            self.role_session = OnlineRoleInferenceSession.from_checkpoint(
                model_path=model_path,
                project_root=config.project_root,
            )
            self.logger.info(
                "Online position-role session loaded for tracking: %s",
                model_path,
            )
        except Exception as exc:
            self.logger.warning(
                "Se desactiva online special-seed role assignment: no se pudo cargar el checkpoint (%s)",
                exc,
            )
            self.enabled = False

    def _reset_runtime_state(self):
        self.prev_positions = {}
        self.last_team_by_special_id = {}
        self.identity_state_by_track_id = {}
        self.raw_frame_prediction_rows = []
        self.segment_assignment_rows = []
        self.stats = {
            "processed_frames": 0,
            "frames_with_role_predictions": 0,
            "position_role_frame_predictions": 0,
            "position_role_player_predictions": 0,
            "identity_segment_resets": 0,
            "segment_level_assignments": 0,
            "special_seed_assigned_frames": 0,
            "special_seed_carry_frames": 0,
            "special_seed_unassigned_frames": 0,
            "special_ids": [int(track_id) for track_id in self.special_ids],
        }

    def reset(self):
        self._reset_runtime_state()

    def _build_segment_expected_slots_by_team(self):
        if self.lineup_matcher is None:
            return dict(self.expected_roles_by_team)
        result = {}
        for team in self.lineup_matcher.lineup_spec.get("teams", []):
            team_name = str(team["team_name"])
            result[team_name] = [
                normalize_slot_token(slot)
                for slot in list(team.get("ui_slots", []) or [])
                if normalize_slot_token(slot)
            ]
        return result or dict(self.expected_roles_by_team)

    def _build_pitch_layout_by_team(self):
        if self.lineup_matcher is None:
            return {}
        result = {}
        for team in self.lineup_matcher.lineup_spec.get("teams", []):
            team_name = str(team["team_name"])
            layout = {}
            for slot, coords in dict(team.get("pitch_layout", {}) or {}).items():
                layout[normalize_slot_token(slot)] = dict(coords)
            result[team_name] = layout
        return result

    def _empty_segment_state(self, track_id, segment_id, frame_id, team_id=None):
        return {
            "track_id": int(track_id),
            "segment_id": int(segment_id),
            "team_id": None if team_id is None else str(team_id),
            "start_frame": int(frame_id),
            "end_frame": int(frame_id),
            "observations": 0,
            "frame_vote_counts": Counter(),
            "frame_vote_confidence_sums": {},
            "model_role_counts": Counter(),
            "recent_positions_m": deque(maxlen=self.recent_window),
            "recent_votes": deque(maxlen=self.recent_window),
            "sum_x_norm": 0.0,
            "sum_y_norm": 0.0,
            "display_role_slot": None,
            "expected_role_slot": None,
            "lineup_slot": None,
            "player_name": None,
            "majority_role": None,
            "majority_expected_role_slot": None,
            "last_position_m": None,
            "reset_reason": None,
        }

    def _ensure_active_segment(self, track_id, frame_id, team_id=None):
        track_key = int(track_id)
        state = self.identity_state_by_track_id.get(track_key)
        if isinstance(state, dict):
            if team_id is not None:
                state["team_id"] = str(team_id)
            return state
        created = self._empty_segment_state(
            track_id=track_key,
            segment_id=1,
            frame_id=frame_id,
            team_id=team_id,
        )
        self.identity_state_by_track_id[track_key] = created
        return created

    def _start_new_segment(self, track_id, frame_id, team_id=None, reason=None):
        previous = self._ensure_active_segment(track_id, frame_id, team_id=team_id)
        new_segment = self._empty_segment_state(
            track_id=int(track_id),
            segment_id=int(previous["segment_id"]) + 1,
            frame_id=frame_id,
            team_id=team_id,
        )
        new_segment["reset_reason"] = reason
        self.identity_state_by_track_id[int(track_id)] = new_segment
        self.stats["identity_segment_resets"] += 1
        return new_segment

    def _should_rotate_segment(self, state, frame_slot, current_position_m, track_data):
        if not isinstance(state, dict):
            return False, None
        if int(state.get("observations", 0)) < int(self.segment_min_observations):
            return False, None
        if current_position_m is None:
            return False, None

        assignment_mode = str(track_data.get("canonical_assignment_mode") or "").strip()
        canonical_relinked = bool(track_data.get("canonical_relinked"))
        if canonical_relinked or assignment_mode in {
            "forced_absorption",
            "canonical_relinked_to_new_raw_tracker",
            "raw_tracker_reassigned_to_other_canonical",
        }:
            return True, "canonical_relinked"

        previous_position_m = state.get("last_position_m")
        if previous_position_m is None:
            return False, None
        jump_m = math.dist(previous_position_m, current_position_m)
        if jump_m < float(self.segment_switch_distance_m):
            return False, None

        majority_slot = state.get("majority_expected_role_slot") or state.get("majority_role")
        if normalize_slot_token(majority_slot) == normalize_slot_token(frame_slot):
            return False, None
        return True, "position_jump_plus_role_change"

    def _build_observations_df(self, tracks_frame, frame_id):
        rows = []
        frame_positions = {}
        for class_name in ("player", "goalkeeper"):
            frame_tracks = tracks_frame.get(class_name, {})
            if not isinstance(frame_tracks, dict):
                continue
            for track_id_raw, track_data in frame_tracks.items():
                if not isinstance(track_data, dict):
                    continue
                try:
                    track_id = int(track_id_raw)
                except (TypeError, ValueError):
                    continue
                if track_id in self.special_ids:
                    continue
                team_id = track_data.get("team")
                if team_id is None:
                    continue
                field_position_m = _safe_field_position_m(track_data)
                if field_position_m is None:
                    continue
                x_m, y_m = field_position_m
                x_norm = float(np.clip(x_m / 106.0, 0.0, 1.0))
                y_norm = float(np.clip(y_m / 68.0, 0.0, 1.0))
                previous = self.prev_positions.get((str(team_id), int(track_id)))
                if previous is None:
                    vx = 0.0
                    vy = 0.0
                else:
                    vx = float(x_norm - previous["x"])
                    vy = float(y_norm - previous["y"])
                rows.append(
                    {
                        "match_id": self.video_path.stem.replace(" ", "_"),
                        "frame_id": int(frame_id),
                        "team_id": str(team_id),
                        "player_id": int(track_id),
                        "class_name": str(class_name),
                        "x_m": float(x_m),
                        "y_m": float(y_m),
                        "x": x_norm,
                        "y": y_norm,
                        "visible": 1,
                        "confidence_tracking": float(track_data.get("confidence", np.nan)),
                        "bbox": track_data.get("bbox"),
                        "role_label": pd.NA,
                        "vx": vx,
                        "vy": vy,
                        "is_interpolated": 0,
                        "role_inference_target": 1,
                    }
                )
                frame_positions[(str(team_id), int(track_id))] = {
                    "x": x_norm,
                    "y": y_norm,
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "frame_id": int(frame_id),
                }
        if not rows:
            return pd.DataFrame(), frame_positions
        return pd.DataFrame(rows), frame_positions

    def _segment_majority(self, state):
        votes = state.get("frame_vote_counts", Counter())
        if not votes:
            return None, 0.0
        best_slot, count = max(votes.items(), key=lambda item: (int(item[1]), str(item[0])))
        conf_sum = float(state.get("frame_vote_confidence_sums", {}).get(best_slot, 0.0))
        mean_conf = conf_sum / max(1, int(count))
        return str(best_slot), float(mean_conf)

    def _update_segment(self, track_id, frame_id, team_id, frame_slot, frame_confidence, row, track_data):
        current_position_m = _safe_field_position_m(track_data)
        state = self._ensure_active_segment(track_id, frame_id, team_id=team_id)
        rotate, reason = self._should_rotate_segment(
            state,
            frame_slot=frame_slot,
            current_position_m=current_position_m,
            track_data=track_data,
        )
        if rotate:
            state = self._start_new_segment(
                track_id=track_id,
                frame_id=frame_id,
                team_id=team_id,
                reason=reason,
            )

        state["team_id"] = str(team_id)
        state["end_frame"] = int(frame_id)
        state["observations"] = int(state["observations"]) + 1
        slot_key = normalize_slot_token(frame_slot)
        state["frame_vote_counts"][slot_key] += 1
        state["frame_vote_confidence_sums"][slot_key] = float(
            state["frame_vote_confidence_sums"].get(slot_key, 0.0)
        ) + float(frame_confidence)
        model_role = normalize_slot_token(
            getattr(row, "predicted_role_unconstrained", None)
            or getattr(row, "matched_model_role", None)
            or getattr(row, "predicted_role", None)
            or frame_slot
        )
        state["model_role_counts"][model_role] += 1
        x_norm = pd.to_numeric(getattr(row, "x", np.nan), errors="coerce")
        y_norm = pd.to_numeric(getattr(row, "y", np.nan), errors="coerce")
        if pd.notna(x_norm) and pd.notna(y_norm):
            state["sum_x_norm"] = float(state["sum_x_norm"]) + float(x_norm)
            state["sum_y_norm"] = float(state["sum_y_norm"]) + float(y_norm)
        if current_position_m is not None:
            state["last_position_m"] = current_position_m
            state["recent_positions_m"].append(current_position_m)
        state["recent_votes"].append(slot_key)
        state["majority_role"] = max(
            state["model_role_counts"].items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )[0]
        state["majority_expected_role_slot"], _ = self._segment_majority(state)
        return state

    def _segment_cost(self, team_id, state, slot_name):
        slot_name = normalize_slot_token(slot_name)
        base_slot = base_role_token(slot_name)
        majority_slot = normalize_slot_token(state.get("majority_expected_role_slot"))
        count = int(state.get("frame_vote_counts", {}).get(base_slot, 0))
        if count == 0:
            count = int(state.get("frame_vote_counts", {}).get(slot_name, 0))
        observations = max(1, int(state.get("observations", 0)))
        ratio = float(count) / float(observations)
        conf_sum = float(state.get("frame_vote_confidence_sums", {}).get(base_slot, 0.0))
        if conf_sum <= 0.0:
            conf_sum = float(state.get("frame_vote_confidence_sums", {}).get(slot_name, 0.0))
        mean_conf = conf_sum / max(1, count)
        cost = 1.0 - ratio - (SEGMENT_CONFIDENCE_WEIGHT * mean_conf)
        if majority_slot and majority_slot != slot_name and base_role_token(majority_slot) != base_slot:
            cost += 1.0

        anchor = _segment_anchor_for_slot(self.pitch_layout_by_team, team_id, slot_name)
        if anchor is not None and observations > 0:
            x_mean = float(state.get("sum_x_norm", 0.0)) / float(observations)
            y_mean = float(state.get("sum_y_norm", 0.0)) / float(observations)
            dx = float(x_mean) - float(anchor[0])
            dy = float(y_mean) - float(anchor[1])
            cost += SEGMENT_POSITION_WEIGHT * math.sqrt((dx * dx) + (dy * dy))
        return float(cost)

    def _assign_segments_for_team(self, team_id, segment_states, expected_slots):
        if not segment_states or not expected_slots:
            return []
        cost_matrix = np.zeros((len(segment_states), len(expected_slots)), dtype=np.float64)
        for row_idx, state in enumerate(segment_states):
            for col_idx, slot_name in enumerate(expected_slots):
                cost_matrix[row_idx, col_idx] = self._segment_cost(team_id, state, slot_name)
        row_ind, col_ind = _solve_assignment(cost_matrix)
        assignments = []
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            assignments.append(
                {
                    "state": segment_states[row_idx],
                    "slot": normalize_slot_token(expected_slots[col_idx]),
                    "cost": float(cost_matrix[row_idx, col_idx]),
                }
            )
        return assignments

    def _resolve_lineup_assignment(self, team_id, state):
        if self.lineup_matcher is None or not isinstance(state, dict):
            return None, None
        return self.lineup_matcher.resolve_player_name(
            team_id,
            state.get("display_role_slot"),
            state.get("majority_expected_role_slot"),
            state.get("majority_role"),
        )

    def _apply_segment_assignment_to_track(self, track_data, state, slot_name, frame_id, assignment_cost):
        frame_conf = float(
            state.get("frame_vote_confidence_sums", {}).get(base_role_token(slot_name), 0.0)
        ) / max(1, int(state.get("frame_vote_counts", {}).get(base_role_token(slot_name), 0)))
        if not np.isfinite(frame_conf):
            frame_conf = 0.0
        track_data["predicted_role"] = str(slot_name)
        track_data["predicted_role_confidence"] = float(frame_conf)
        track_data["display_role_slot"] = str(slot_name)
        track_data["expected_role_slot"] = str(slot_name)
        track_data["assignment_method"] = "hungarian_segment"
        track_data["assignment_cost"] = float(assignment_cost)
        track_data["stable_role_assignment_method"] = "hungarian_segment"
        track_data["role_stabilized"] = True
        track_data["role_stabilized_at_frame"] = int(frame_id)
        track_data["role_stabilization_observations"] = int(state.get("observations", 0))
        track_data["identity_segment_id"] = int(state.get("segment_id", 1))
        track_data["identity_segment_start_frame"] = int(state.get("start_frame", frame_id))
        track_data["identity_segment_observations"] = int(state.get("observations", 0))
        track_data["segment_majority_role"] = state.get("majority_role")
        track_data["segment_majority_expected_role_slot"] = state.get("majority_expected_role_slot")
        track_data["identity_reset_reason"] = state.get("reset_reason")

        state["display_role_slot"] = str(slot_name)
        state["expected_role_slot"] = str(slot_name)
        resolved_slot, player_name = self._resolve_lineup_assignment(track_data.get("team"), state)
        if resolved_slot and player_name:
            track_data["lineup_slot"] = str(resolved_slot)
            track_data["player_name"] = str(player_name)
            state["lineup_slot"] = str(resolved_slot)
            state["player_name"] = str(player_name)

    def _assign_special_seed_frame_teams(self, tracks_frame, visible_player_df):
        defender_candidates = []
        for row in visible_player_df.itertuples(index=False):
            slot_name = normalize_slot_token(
                getattr(row, "expected_role_slot", None) or getattr(row, "predicted_role", None)
            )
            if base_role_token(slot_name) not in self.defender_roles and slot_name not in self.defender_roles:
                continue
            x_m = pd.to_numeric(getattr(row, "x_m", np.nan), errors="coerce")
            y_m = pd.to_numeric(getattr(row, "y_m", np.nan), errors="coerce")
            if pd.isna(x_m) or pd.isna(y_m):
                continue
            defender_candidates.append(
                {
                    "track_id": int(row.player_id),
                    "team": str(row.team_id),
                    "position": (float(x_m), float(y_m)),
                }
            )

        for class_name in ("player", "goalkeeper"):
            frame_tracks = tracks_frame.get(class_name, {})
            if not isinstance(frame_tracks, dict):
                continue
            for special_id in self.special_ids:
                for track_key in _special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if not isinstance(track_data, dict):
                        continue
                    field_position = _safe_field_position_m(track_data)
                    if field_position is None:
                        continue
                    best_candidate = None
                    best_distance_m = None
                    for defender in defender_candidates:
                        distance_m = math.dist(field_position, defender["position"])
                        if best_distance_m is None or distance_m < best_distance_m:
                            best_distance_m = distance_m
                            best_candidate = defender
                    if best_candidate is not None:
                        resolved_team = str(best_candidate["team"])
                        track_data["team"] = resolved_team
                        track_data["special_team_assignment_source"] = "nearest_defender_role"
                        track_data["special_team_assignment_distance_m"] = float(best_distance_m)
                        track_data["special_team_assignment_track_id"] = int(best_candidate["track_id"])
                        self.last_team_by_special_id[int(special_id)] = resolved_team
                        self.stats["special_seed_assigned_frames"] += 1
                    elif int(special_id) in self.last_team_by_special_id:
                        track_data["team"] = self.last_team_by_special_id[int(special_id)]
                        track_data["special_team_assignment_source"] = "nearest_defender_role_carry"
                        track_data["special_team_assignment_distance_m"] = None
                        track_data["special_team_assignment_track_id"] = None
                        self.stats["special_seed_carry_frames"] += 1
                    else:
                        track_data["special_team_assignment_source"] = "unassigned"
                        track_data["special_team_assignment_distance_m"] = None
                        track_data["special_team_assignment_track_id"] = None
                        self.stats["special_seed_unassigned_frames"] += 1

    def _annotate_special_goalkeepers(self, tracks_frame, frame_id):
        for class_name in ("player", "goalkeeper"):
            frame_tracks = tracks_frame.get(class_name, {})
            if not isinstance(frame_tracks, dict):
                continue
            for special_id in self.special_ids:
                for track_key in _special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if not isinstance(track_data, dict):
                        continue
                    track_data["predicted_role_frame"] = "POR"
                    track_data["predicted_role_frame_confidence"] = 1.0
                    track_data["predicted_role"] = "POR"
                    track_data["predicted_role_confidence"] = 1.0
                    track_data["display_role_slot"] = "POR"
                    track_data["expected_role_slot"] = "POR"
                    track_data["assignment_method"] = "manual_special_goalkeeper"
                    track_data["stable_role_assignment_method"] = "manual_special_goalkeeper"
                    track_data["role_stabilized"] = True
                    track_data["role_stabilized_at_frame"] = int(frame_id)

    def process_tracks_frame(self, tracks_frame, frame_id):
        self.stats["processed_frames"] += 1
        if not self.enabled or self.role_session is None:
            return {
                "frame_predictions": 0,
                "player_predictions": 0,
                "roles_applied": False,
            }

        observations_df, frame_positions = self._build_observations_df(tracks_frame, frame_id)
        if observations_df.empty:
            self._annotate_special_goalkeepers(tracks_frame, frame_id)
            return {
                "frame_predictions": 0,
                "player_predictions": 0,
                "roles_applied": False,
            }

        role_result = self.role_session.predict_frame(
            observations_df,
            expected_roles_by_team=self.expected_roles_by_team,
            include_all_targets=True,
        )
        frame_predictions_df = role_result.get("frame_predictions_df", pd.DataFrame()).copy()
        player_predictions_df = role_result.get("player_predictions_df", pd.DataFrame()).copy()

        if frame_predictions_df.empty or player_predictions_df.empty:
            self._annotate_special_goalkeepers(tracks_frame, frame_id)
            return {
                "frame_predictions": 0,
                "player_predictions": 0,
                "roles_applied": False,
            }

        visible_player_df = player_predictions_df.copy()
        self._assign_special_seed_frame_teams(tracks_frame, visible_player_df)

        active_segments_by_team = {}
        roles_applied = False
        for row in visible_player_df.itertuples(index=False):
            team_id = str(row.team_id)
            track_id = int(row.player_id)
            class_name, track_key, track_data = _first_existing_track_payload(tracks_frame, track_id)
            if track_data is None:
                continue

            frame_slot = normalize_slot_token(
                getattr(row, "expected_role_slot", None) or getattr(row, "predicted_role", None)
            )
            frame_conf = float(
                pd.to_numeric(getattr(row, "predicted_role_confidence", np.nan), errors="coerce")
            )
            if not np.isfinite(frame_conf):
                frame_conf = 0.0
            track_data["predicted_role_frame"] = str(frame_slot)
            track_data["predicted_role_frame_confidence"] = float(frame_conf)
            if hasattr(row, "predicted_role_unconstrained"):
                track_data["predicted_role_unconstrained"] = str(row.predicted_role_unconstrained)
            if hasattr(row, "predicted_role_confidence_unconstrained"):
                unconstrained_conf = pd.to_numeric(
                    row.predicted_role_confidence_unconstrained,
                    errors="coerce",
                )
                if pd.notna(unconstrained_conf):
                    track_data["predicted_role_confidence_unconstrained"] = float(unconstrained_conf)
            if hasattr(row, "matched_model_role") and pd.notna(row.matched_model_role):
                track_data["matched_model_role"] = str(row.matched_model_role)
            if hasattr(row, "assignment_cost") and pd.notna(row.assignment_cost):
                track_data["assignment_cost"] = float(row.assignment_cost)
            track_data["assignment_method"] = "hungarian_frame"

            state = self._update_segment(
                track_id=track_id,
                frame_id=frame_id,
                team_id=team_id,
                frame_slot=frame_slot,
                frame_confidence=frame_conf,
                row=row,
                track_data=track_data,
            )
            active_segments_by_team.setdefault(team_id, []).append((state, track_data))
            roles_applied = True

            self.raw_frame_prediction_rows.append(
                {
                    "frame_id": int(frame_id),
                    "team_id": str(team_id),
                    "player_id": int(track_id),
                    "class_name": str(class_name),
                    "identity_segment_id": int(state.get("segment_id", 1)),
                    "predicted_role_frame": str(frame_slot),
                    "predicted_role_frame_confidence": float(frame_conf),
                    "predicted_role_unconstrained": track_data.get("predicted_role_unconstrained"),
                    "predicted_role_confidence_unconstrained": track_data.get(
                        "predicted_role_confidence_unconstrained"
                    ),
                    "matched_model_role": track_data.get("matched_model_role"),
                    "x_m": float(pd.to_numeric(getattr(row, "x_m", np.nan), errors="coerce")),
                    "y_m": float(pd.to_numeric(getattr(row, "y_m", np.nan), errors="coerce")),
                    "x": float(pd.to_numeric(getattr(row, "x", np.nan), errors="coerce")),
                    "y": float(pd.to_numeric(getattr(row, "y", np.nan), errors="coerce")),
                }
            )

        for team_id, state_pairs in active_segments_by_team.items():
            expected_slots = self.segment_expected_slots_by_team.get(
                str(team_id),
                self.expected_roles_by_team.get(str(team_id), []),
            )
            assignments = self._assign_segments_for_team(
                team_id=str(team_id),
                segment_states=[state for state, _ in state_pairs],
                expected_slots=expected_slots,
            )
            assigned_by_segment = {
                int(item["state"]["segment_id"]): item for item in assignments
            }
            for state, track_data in state_pairs:
                assigned = assigned_by_segment.get(int(state.get("segment_id", -1)))
                final_slot = (
                    assigned["slot"]
                    if assigned is not None
                    else state.get("majority_expected_role_slot")
                    or state.get("majority_role")
                )
                assignment_cost = float(assigned["cost"]) if assigned is not None else math.nan
                self._apply_segment_assignment_to_track(
                    track_data,
                    state,
                    slot_name=final_slot,
                    frame_id=frame_id,
                    assignment_cost=assignment_cost,
                )
                self.stats["segment_level_assignments"] += 1
                self.segment_assignment_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "team_id": str(team_id),
                        "player_id": int(state["track_id"]),
                        "identity_segment_id": int(state["segment_id"]),
                        "display_role_slot": str(final_slot),
                        "segment_majority_role": state.get("majority_role"),
                        "segment_majority_expected_role_slot": state.get(
                            "majority_expected_role_slot"
                        ),
                        "assignment_cost": assignment_cost,
                        "observations": int(state.get("observations", 0)),
                        "player_name": state.get("player_name"),
                        "lineup_slot": state.get("lineup_slot"),
                    }
                )

        self._annotate_special_goalkeepers(tracks_frame, frame_id)
        self.prev_positions.update(frame_positions)

        frame_predictions = int(len(frame_predictions_df))
        player_predictions = int(
            visible_player_df[["team_id", "player_id"]].drop_duplicates().shape[0]
        )
        self.stats["position_role_frame_predictions"] += frame_predictions
        self.stats["position_role_player_predictions"] += player_predictions
        if roles_applied:
            self.stats["frames_with_role_predictions"] += 1

        return {
            "frame_predictions": frame_predictions,
            "player_predictions": player_predictions,
            "roles_applied": bool(roles_applied),
        }

    def build_role_export_dataframes(self, tracks):
        frame_df = pd.DataFrame(self.raw_frame_prediction_rows)
        if not frame_df.empty:
            frame_df = frame_df.sort_values(
                ["frame_id", "team_id", "player_id"]
            ).reset_index(drop=True)

        player_rows = []
        for track_id, state in sorted(self.identity_state_by_track_id.items()):
            player_rows.append(
                {
                    "team_id": state.get("team_id"),
                    "player_id": int(track_id),
                    "identity_segment_id": int(state.get("segment_id", 1)),
                    "first_frame_id": int(state.get("start_frame", 0)),
                    "last_frame_id": int(state.get("end_frame", 0)),
                    "frames_seen": int(state.get("observations", 0)),
                    "predicted_role": state.get("display_role_slot"),
                    "display_role_slot": state.get("display_role_slot"),
                    "expected_role_slot": state.get("expected_role_slot"),
                    "segment_majority_role": state.get("majority_role"),
                    "segment_majority_expected_role_slot": state.get(
                        "majority_expected_role_slot"
                    ),
                    "player_name": state.get("player_name"),
                    "lineup_slot": state.get("lineup_slot"),
                    "reset_reason": state.get("reset_reason"),
                }
            )
        player_df = pd.DataFrame(player_rows)
        if not player_df.empty:
            player_df = player_df.sort_values(
                ["team_id", "player_id", "identity_segment_id"]
            ).reset_index(drop=True)

        greedy_df = pd.DataFrame(self.segment_assignment_rows)
        if not greedy_df.empty:
            greedy_df = greedy_df.sort_values(
                ["frame_id", "team_id", "player_id"]
            ).reset_index(drop=True)
        return frame_df, player_df, greedy_df

    def on_frame(self, tracks, frame_id):
        frame_tracks = {}
        for class_name in ("player", "goalkeeper", "referee", "ball"):
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                frame_tracks[class_name] = {}
                continue
            frame_payload = class_frames[frame_id]
            frame_tracks[class_name] = (
                frame_payload if isinstance(frame_payload, dict) else {}
            )
        self.process_tracks_frame(frame_tracks, frame_id)

    def summary(self):
        return dict(self.stats)


__all__ = [
    "OnlineSpecialSeedRoleAssigner",
    "apply_special_seed_role_team_assignment",
    "build_role_artifacts_output_dir",
    "build_role_predictions_output_paths",
    "copy_output_artifact",
    "save_dataframe_csv",
    "save_role_visualizations",
    "sanitize_video_stem",
]
