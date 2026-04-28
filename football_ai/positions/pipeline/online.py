"""
Motor online de roles posicionales basado en doble pasada de Húngaro.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover
    _scipy_linear_sum_assignment = None
from .config import (
    DEFAULT_SPECIAL_SEED_CANONICAL_IDS,
    DEFAULT_SPECIAL_SEED_DEFENDER_ROLES,
    DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH,
    SEGMENT_CONFIDENCE_WEIGHT,
    SEGMENT_POSITION_WEIGHT,
    normalize_expected_roles_mapping,
    resolve_expected_roles_by_team_from_config,
)
from .helpers import (
    first_existing_track_payload,
    safe_field_position_m,
    segment_anchor_for_slot,
    solve_assignment,
    special_seed_frame_track_keys,
)
from .lineup_spec import LineupSlotMatcher, base_role_token, normalize_slot_token


def _coalesce_slot_value(*values):
    for value in values:
        if value is None or pd.isna(value):
            continue
        token = str(value).strip()
        if token:
            return token
    return None


class OnlineSpecialSeedRoleAssigner:
    def __init__(self, config, video_path, logger, *, expected_roles_by_team_override=None, lineup_matcher=None):
        self.config = config
        self.video_path = Path(video_path)
        self.logger = logger
        tracking_cfg = getattr(config, "tracking", {}) or {}
        self.enabled = bool(tracking_cfg.get("special_seed_role_team_assignment_enabled", True))
        self.special_ids = tuple(int(track_id) for track_id in tracking_cfg.get("special_seed_canonical_ids", list(DEFAULT_SPECIAL_SEED_CANONICAL_IDS)))
        self.defender_roles = {normalize_slot_token(role) for role in tracking_cfg.get("special_seed_defender_roles", list(DEFAULT_SPECIAL_SEED_DEFENDER_ROLES))}
        self.expected_roles_by_team = (
            normalize_expected_roles_mapping(expected_roles_by_team_override)
            if expected_roles_by_team_override is not None
            else resolve_expected_roles_by_team_from_config(config)
        )
        self.lineup_matcher = lineup_matcher if isinstance(lineup_matcher, LineupSlotMatcher) else None
        self.segment_expected_slots_by_team = self._build_segment_expected_slots_by_team()
        self.pitch_layout_by_team = self._build_pitch_layout_by_team()
        self.segment_switch_distance_m = float(tracking_cfg.get("role_segment_switch_distance_m", tracking_cfg.get("role_swap_position_jump_m", 14.0)))
        self.segment_min_observations = max(2, int(tracking_cfg.get("role_segment_min_observations", tracking_cfg.get("role_swap_min_recent_samples", 6))))
        self.recent_window = max(3, int(tracking_cfg.get("role_segment_recent_window", 10)))
        self._reset_runtime_state()
        self.role_session = None
        if not self.enabled:
            return
        model_path = Path(config.project_root) / tracking_cfg.get("special_seed_role_model_path", DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH)
        if not model_path.exists():
            self.logger.warning("Se desactiva online special-seed role assignment: no existe el checkpoint %s", model_path)
            self.enabled = False
            return
        try:
            from ..model.session import OnlineRoleInferenceSession

            self.role_session = OnlineRoleInferenceSession.from_checkpoint(model_path=model_path, project_root=config.project_root)
            self.logger.info("Online position-role session loaded for tracking: %s", model_path)
        except Exception as exc:
            self.logger.warning("Se desactiva online special-seed role assignment: %s", exc)
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
            result[str(team["team_name"])] = [
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
            result[str(team["team_name"])] = {
                normalize_slot_token(slot): dict(coords)
                for slot, coords in dict(team.get("pitch_layout", {}) or {}).items()
            }
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
        state = self.identity_state_by_track_id.get(int(track_id))
        if isinstance(state, dict):
            if team_id is not None:
                state["team_id"] = str(team_id)
            return state
        created = self._empty_segment_state(int(track_id), 1, frame_id, team_id=team_id)
        self.identity_state_by_track_id[int(track_id)] = created
        return created

    def _start_new_segment(self, track_id, frame_id, team_id=None, reason=None):
        previous = self._ensure_active_segment(track_id, frame_id, team_id=team_id)
        new_segment = self._empty_segment_state(int(track_id), int(previous["segment_id"]) + 1, frame_id, team_id=team_id)
        new_segment["reset_reason"] = reason
        self.identity_state_by_track_id[int(track_id)] = new_segment
        self.stats["identity_segment_resets"] += 1
        return new_segment

    def _should_rotate_segment(self, state, frame_slot, current_position_m, track_data):
        if int(state.get("observations", 0)) < int(self.segment_min_observations) or current_position_m is None:
            return False, None
        assignment_mode = str(track_data.get("canonical_assignment_mode") or "").strip()
        if bool(track_data.get("canonical_relinked")) or assignment_mode in {"forced_absorption", "canonical_relinked_to_new_raw_tracker", "raw_tracker_reassigned_to_other_canonical"}:
            return True, "canonical_relinked"
        previous_position_m = state.get("last_position_m")
        if previous_position_m is None or math.dist(previous_position_m, current_position_m) < float(self.segment_switch_distance_m):
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
                field_position_m = safe_field_position_m(track_data)
                if team_id is None or field_position_m is None:
                    continue
                x_m, y_m = field_position_m
                x_norm = float(np.clip(x_m / 106.0, 0.0, 1.0))
                y_norm = float(np.clip(y_m / 68.0, 0.0, 1.0))
                previous = self.prev_positions.get((str(team_id), int(track_id)))
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
                        "vx": 0.0 if previous is None else float(x_norm - previous["x"]),
                        "vy": 0.0 if previous is None else float(y_norm - previous["y"]),
                        "is_interpolated": 0,
                        "role_inference_target": 1,
                    }
                )
                frame_positions[(str(team_id), int(track_id))] = {"x": x_norm, "y": y_norm, "x_m": float(x_m), "y_m": float(y_m)}
        return (pd.DataFrame(rows), frame_positions) if rows else (pd.DataFrame(), frame_positions)

    def _segment_majority(self, state):
        votes = state.get("frame_vote_counts", Counter())
        if not votes:
            return None, 0.0
        best_slot, count = max(votes.items(), key=lambda item: (int(item[1]), str(item[0])))
        mean_conf = float(state.get("frame_vote_confidence_sums", {}).get(best_slot, 0.0)) / max(1, int(count))
        return str(best_slot), float(mean_conf)

    def _update_segment(self, track_id, frame_id, team_id, frame_slot, frame_confidence, row, track_data):
        current_position_m = safe_field_position_m(track_data)
        state = self._ensure_active_segment(track_id, frame_id, team_id=team_id)
        rotate, reason = self._should_rotate_segment(state, frame_slot, current_position_m, track_data)
        if rotate:
            state = self._start_new_segment(track_id, frame_id, team_id=team_id, reason=reason)
        slot_key = normalize_slot_token(frame_slot)
        state["team_id"] = str(team_id)
        state["end_frame"] = int(frame_id)
        state["observations"] += 1
        state["frame_vote_counts"][slot_key] += 1
        state["frame_vote_confidence_sums"][slot_key] = float(state["frame_vote_confidence_sums"].get(slot_key, 0.0)) + float(frame_confidence)
        model_role = normalize_slot_token(
            getattr(row, "predicted_role_unconstrained", None) or getattr(row, "matched_model_role", None) or getattr(row, "predicted_role", None) or frame_slot
        )
        state["model_role_counts"][model_role] += 1
        x_norm = pd.to_numeric(getattr(row, "x", np.nan), errors="coerce")
        y_norm = pd.to_numeric(getattr(row, "y", np.nan), errors="coerce")
        if pd.notna(x_norm) and pd.notna(y_norm):
            state["sum_x_norm"] += float(x_norm)
            state["sum_y_norm"] += float(y_norm)
        if current_position_m is not None:
            state["last_position_m"] = current_position_m
            state["recent_positions_m"].append(current_position_m)
        state["recent_votes"].append(slot_key)
        state["majority_role"] = max(state["model_role_counts"].items(), key=lambda item: (int(item[1]), str(item[0])))[0]
        state["majority_expected_role_slot"], _ = self._segment_majority(state)
        return state

    def _segment_cost(self, team_id, state, slot_name):
        slot_name = normalize_slot_token(slot_name)
        base_slot = base_role_token(slot_name)
        count = int(state.get("frame_vote_counts", {}).get(base_slot, 0)) or int(state.get("frame_vote_counts", {}).get(slot_name, 0))
        observations = max(1, int(state.get("observations", 0)))
        conf_sum = float(state.get("frame_vote_confidence_sums", {}).get(base_slot, 0.0)) or float(state.get("frame_vote_confidence_sums", {}).get(slot_name, 0.0))
        cost = 1.0 - (float(count) / float(observations)) - (SEGMENT_CONFIDENCE_WEIGHT * (conf_sum / max(1, count)))
        majority_slot = normalize_slot_token(state.get("majority_expected_role_slot"))
        if majority_slot and majority_slot != slot_name and base_role_token(majority_slot) != base_slot:
            cost += 1.0
        anchor = segment_anchor_for_slot(self.pitch_layout_by_team, team_id, slot_name)
        if anchor is not None and observations > 0:
            x_mean = float(state.get("sum_x_norm", 0.0)) / float(observations)
            y_mean = float(state.get("sum_y_norm", 0.0)) / float(observations)
            cost += SEGMENT_POSITION_WEIGHT * math.sqrt(((x_mean - float(anchor[0])) ** 2) + ((y_mean - float(anchor[1])) ** 2))
        return float(cost)

    def _assign_segments_for_team(self, team_id, segment_states, expected_slots):
        if not segment_states or not expected_slots:
            return []
        cost_matrix = np.zeros((len(segment_states), len(expected_slots)), dtype=np.float64)
        for row_idx, state in enumerate(segment_states):
            for col_idx, slot_name in enumerate(expected_slots):
                cost_matrix[row_idx, col_idx] = self._segment_cost(team_id, state, slot_name)
        row_ind, col_ind = solve_assignment(cost_matrix, _scipy_linear_sum_assignment)
        return [{"state": segment_states[row_idx], "slot": normalize_slot_token(expected_slots[col_idx]), "cost": float(cost_matrix[row_idx, col_idx])} for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist())]

    def _resolve_lineup_assignment(self, team_id, state):
        if self.lineup_matcher is None:
            return None, None
        return self.lineup_matcher.resolve_player_name(team_id, state.get("display_role_slot"), state.get("majority_expected_role_slot"), state.get("majority_role"))

    def _apply_segment_assignment_to_track(self, track_data, state, slot_name, frame_id, assignment_cost):
        base_slot = base_role_token(slot_name)
        frame_conf = float(state.get("frame_vote_confidence_sums", {}).get(base_slot, 0.0)) / max(1, int(state.get("frame_vote_counts", {}).get(base_slot, 0)))
        if not np.isfinite(frame_conf):
            frame_conf = 0.0
        track_data.update(
            {
                "predicted_role": str(slot_name),
                "predicted_role_confidence": float(frame_conf),
                "display_role_slot": str(slot_name),
                "expected_role_slot": str(slot_name),
                "assignment_method": "hungarian_segment",
                "assignment_cost": float(assignment_cost),
                "stable_role_assignment_method": "hungarian_segment",
                "role_stabilized": True,
                "role_stabilized_at_frame": int(frame_id),
                "role_stabilization_observations": int(state.get("observations", 0)),
                "identity_segment_id": int(state.get("segment_id", 1)),
                "identity_segment_start_frame": int(state.get("start_frame", frame_id)),
                "identity_segment_observations": int(state.get("observations", 0)),
                "segment_majority_role": state.get("majority_role"),
                "segment_majority_expected_role_slot": state.get("majority_expected_role_slot"),
                "identity_reset_reason": state.get("reset_reason"),
            }
        )
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
            raw_slot = _coalesce_slot_value(
                getattr(row, "expected_role_slot", None),
                getattr(row, "predicted_role", None),
            )
            slot_name = normalize_slot_token(raw_slot)
            if base_role_token(slot_name) not in self.defender_roles and slot_name not in self.defender_roles:
                continue
            x_m = pd.to_numeric(getattr(row, "x_m", np.nan), errors="coerce")
            y_m = pd.to_numeric(getattr(row, "y_m", np.nan), errors="coerce")
            if pd.notna(x_m) and pd.notna(y_m):
                defender_candidates.append({"track_id": int(row.player_id), "team": str(row.team_id), "position": (float(x_m), float(y_m))})
        for class_name in ("player", "goalkeeper"):
            frame_tracks = tracks_frame.get(class_name, {})
            if not isinstance(frame_tracks, dict):
                continue
            for special_id in self.special_ids:
                for track_key in special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if not isinstance(track_data, dict):
                        continue
                    field_position = safe_field_position_m(track_data)
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
                for track_key in special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if isinstance(track_data, dict):
                        track_data.update(
                            {
                                "predicted_role_frame": "POR",
                                "predicted_role_frame_confidence": 1.0,
                                "predicted_role": "POR",
                                "predicted_role_confidence": 1.0,
                                "display_role_slot": "POR",
                                "expected_role_slot": "POR",
                                "assignment_method": "manual_special_goalkeeper",
                                "stable_role_assignment_method": "manual_special_goalkeeper",
                                "role_stabilized": True,
                                "role_stabilized_at_frame": int(frame_id),
                            }
                        )

    def process_tracks_frame(self, tracks_frame, frame_id):
        self.stats["processed_frames"] += 1
        if not self.enabled or self.role_session is None:
            return {"frame_predictions": 0, "player_predictions": 0, "roles_applied": False}
        observations_df, frame_positions = self._build_observations_df(tracks_frame, frame_id)
        if observations_df.empty:
            self._annotate_special_goalkeepers(tracks_frame, frame_id)
            return {"frame_predictions": 0, "player_predictions": 0, "roles_applied": False}
        role_result = self.role_session.predict_frame(observations_df, expected_roles_by_team=self.expected_roles_by_team, include_all_targets=True)
        frame_predictions_df = role_result.get("frame_predictions_df", pd.DataFrame()).copy()
        player_predictions_df = role_result.get("player_predictions_df", pd.DataFrame()).copy()
        if frame_predictions_df.empty or player_predictions_df.empty:
            self._annotate_special_goalkeepers(tracks_frame, frame_id)
            return {"frame_predictions": 0, "player_predictions": 0, "roles_applied": False}

        visible_player_df = player_predictions_df.copy()
        self._assign_special_seed_frame_teams(tracks_frame, visible_player_df)
        active_segments_by_team = {}
        roles_applied = False
        for row in visible_player_df.itertuples(index=False):
            team_id = str(row.team_id)
            track_id = int(row.player_id)
            class_name, _, track_data = first_existing_track_payload(tracks_frame, track_id)
            if track_data is None:
                continue
            raw_slot = _coalesce_slot_value(
                getattr(row, "expected_role_slot", None),
                getattr(row, "predicted_role", None),
            )
            frame_slot = normalize_slot_token(raw_slot)
            frame_conf = float(pd.to_numeric(getattr(row, "predicted_role_confidence", np.nan), errors="coerce"))
            if not np.isfinite(frame_conf):
                frame_conf = 0.0
            track_data["predicted_role_frame"] = str(frame_slot)
            track_data["predicted_role_frame_confidence"] = float(frame_conf)
            for attr in ("predicted_role_unconstrained", "predicted_role_confidence_unconstrained", "matched_model_role", "assignment_cost"):
                if hasattr(row, attr) and pd.notna(getattr(row, attr)):
                    value = getattr(row, attr)
                    track_data[attr] = float(value) if "confidence" in attr or attr == "assignment_cost" else str(value)
            track_data["assignment_method"] = "hungarian_frame"
            state = self._update_segment(track_id, frame_id, team_id, frame_slot, frame_conf, row, track_data)
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
                    "predicted_role_confidence_unconstrained": track_data.get("predicted_role_confidence_unconstrained"),
                    "matched_model_role": track_data.get("matched_model_role"),
                    "x_m": float(pd.to_numeric(getattr(row, "x_m", np.nan), errors="coerce")),
                    "y_m": float(pd.to_numeric(getattr(row, "y_m", np.nan), errors="coerce")),
                    "x": float(pd.to_numeric(getattr(row, "x", np.nan), errors="coerce")),
                    "y": float(pd.to_numeric(getattr(row, "y", np.nan), errors="coerce")),
                }
            )

        for team_id, state_pairs in active_segments_by_team.items():
            expected_slots = self.segment_expected_slots_by_team.get(str(team_id), self.expected_roles_by_team.get(str(team_id), []))
            assignments = self._assign_segments_for_team(str(team_id), [state for state, _ in state_pairs], expected_slots)
            assigned_by_segment = {int(item["state"]["segment_id"]): item for item in assignments}
            for state, track_data in state_pairs:
                assigned = assigned_by_segment.get(int(state.get("segment_id", -1)))
                final_slot = assigned["slot"] if assigned is not None else state.get("majority_expected_role_slot") or state.get("majority_role")
                assignment_cost = float(assigned["cost"]) if assigned is not None else math.nan
                self._apply_segment_assignment_to_track(track_data, state, final_slot, frame_id, assignment_cost)
                self.stats["segment_level_assignments"] += 1
                self.segment_assignment_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "team_id": str(team_id),
                        "player_id": int(state["track_id"]),
                        "identity_segment_id": int(state["segment_id"]),
                        "display_role_slot": str(final_slot),
                        "segment_majority_role": state.get("majority_role"),
                        "segment_majority_expected_role_slot": state.get("majority_expected_role_slot"),
                        "assignment_cost": assignment_cost,
                        "observations": int(state.get("observations", 0)),
                        "player_name": state.get("player_name"),
                        "lineup_slot": state.get("lineup_slot"),
                    }
                )

        self._annotate_special_goalkeepers(tracks_frame, frame_id)
        self.prev_positions.update(frame_positions)
        frame_predictions = int(len(frame_predictions_df))
        player_predictions = int(visible_player_df[["team_id", "player_id"]].drop_duplicates().shape[0])
        self.stats["position_role_frame_predictions"] += frame_predictions
        self.stats["position_role_player_predictions"] += player_predictions
        if roles_applied:
            self.stats["frames_with_role_predictions"] += 1
        return {"frame_predictions": frame_predictions, "player_predictions": player_predictions, "roles_applied": bool(roles_applied)}

    def build_role_export_dataframes(self, tracks):
        frame_df = pd.DataFrame(self.raw_frame_prediction_rows)
        if not frame_df.empty:
            frame_df = frame_df.sort_values(["frame_id", "team_id", "player_id"]).reset_index(drop=True)
        player_df = pd.DataFrame(
            [
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
                    "segment_majority_expected_role_slot": state.get("majority_expected_role_slot"),
                    "player_name": state.get("player_name"),
                    "lineup_slot": state.get("lineup_slot"),
                    "reset_reason": state.get("reset_reason"),
                }
                for track_id, state in sorted(self.identity_state_by_track_id.items())
            ]
        )
        if not player_df.empty:
            player_df = player_df.sort_values(["team_id", "player_id", "identity_segment_id"]).reset_index(drop=True)
        segment_df = pd.DataFrame(self.segment_assignment_rows)
        if not segment_df.empty:
            segment_df = segment_df.sort_values(["frame_id", "team_id", "player_id"]).reset_index(drop=True)
        return frame_df, player_df, segment_df

    def on_frame(self, tracks, frame_id):
        frame_tracks = {}
        for class_name in ("player", "goalkeeper", "referee", "ball"):
            class_frames = tracks.get(class_name, [])
            frame_tracks[class_name] = {} if frame_id >= len(class_frames) else (class_frames[frame_id] if isinstance(class_frames[frame_id], dict) else {})
        self.process_tracks_frame(frame_tracks, frame_id)

    def summary(self):
        return dict(self.stats)
