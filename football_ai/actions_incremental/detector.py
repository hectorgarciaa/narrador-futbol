from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from football_ai.pathcrf_slot_mapping import (
    all_person_slots,
    all_referee_slots,
    canonical_id_to_person_slot,
    canonical_id_to_referee_slot,
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)


TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")

HOME_TEAM_TEMPLATE = np.array(
    [
        [6.0, 34.0],
        [18.0, 12.0],
        [18.0, 26.0],
        [18.0, 42.0],
        [18.0, 56.0],
        [42.0, 18.0],
        [42.0, 34.0],
        [42.0, 50.0],
        [74.0, 16.0],
        [78.0, 34.0],
        [74.0, 52.0],
    ],
    dtype=np.float32,
)

REFEREE_TEMPLATE = np.array(
    [
        [30.0, 34.0],
        [52.5, 18.0],
        [75.0, 50.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class ActionsDetectorConfig:
    fps: float = 25.0
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    expected_players_per_team: int = 11
    expected_referees: int = 3
    carrier_max_distance_px: float = 120.0
    bbox_inside_padding_px: float = 10.0
    export_possession_targets: bool = False
    player_outlier_speed_mps: float = 14.0
    referee_outlier_speed_mps: float = 12.0
    ball_outlier_speed_mps: float = 35.0
    smoothing_window: int = 9
    window_size_frames: int | None = None


@dataclass
class TrackObservation:
    frame_id: int
    class_name: str
    team_name: str | None
    predicted_role: str | None
    bbox: tuple[float, float, float, float] | None
    field_position_m: tuple[float, float] | None


@dataclass
class TrackRecord:
    raw_track_id: str
    class_name: str
    observations: list[TrackObservation] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return min(obs.frame_id for obs in self.observations)

    def stable_team_name(self) -> str | None:
        counts: dict[str, int] = {}
        for obs in self.observations:
            if obs.team_name:
                counts[obs.team_name] = counts.get(obs.team_name, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def stable_role(self) -> str | None:
        counts: dict[str, int] = {}
        for obs in self.observations:
            if obs.predicted_role:
                counts[obs.predicted_role] = counts.get(obs.predicted_role, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def median_field_position(self) -> tuple[float, float] | None:
        coords = [obs.field_position_m for obs in self.observations if obs.field_position_m is not None]
        if not coords:
            return None
        arr = np.asarray(coords, dtype=np.float32)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    def median_x(self, fallback: float) -> float:
        field_pos = self.median_field_position()
        if field_pos is not None:
            return float(field_pos[0])
        return float(fallback)


@dataclass
class ActionsDetectorSnapshot:
    frame_count: int
    track_counts_by_class: dict[str, int]
    possession_frames: int
    window_start_frame: int
    window_end_frame: int


@dataclass
class ActionsDetectorSummary:
    track_count_by_class: dict[str, int]
    person_slot_assignments: dict[str, str]
    referee_slot_assignments: dict[str, str]
    synthetic_home_slots: list[str]
    synthetic_away_slots: list[str]
    synthetic_referee_slots: list[str]
    frames: int


@dataclass
class MaterializedSlotState:
    frame_count: int
    person_slot_assignments: dict[str, str]
    referee_slot_assignments: dict[str, str]
    slot_tracks: dict[str, pd.DataFrame]
    referee_tracks: dict[str, pd.DataFrame]
    ball_track: pd.DataFrame
    carrier_series: pd.Series
    owning_team_series: pd.Series
    synthetic_home_slots: list[str]
    synthetic_away_slots: list[str]
    synthetic_referee_slots: list[str]
    tracking_df: pd.DataFrame


class ActionsDetector:
    """Incremental builder of the PathCRF tracking input without depending on the legacy adapter.

    Each `update(frame_n, packet_n)` only ingests the current frame and updates the detector state.
    Materialization uses the same adaptation semantics as the offline path, but implemented locally.
    """

    def __init__(self, config: ActionsDetectorConfig | None = None):
        self.config = config or ActionsDetectorConfig()
        self.reset()

    def reset(self) -> None:
        self._frames_seen = 0
        self._last_frame_index = -1
        self._snapshot: dict[str, list[dict[str, Any]]] = {
            "player": [],
            "goalkeeper": [],
            "referee": [],
            "ball": [],
            "possession": [],
        }
        self._records: dict[str, dict[str, TrackRecord]] = {
            "player": {},
            "goalkeeper": {},
            "referee": {},
            "ball": {},
        }
        self._materialized_state: MaterializedSlotState | None = None
        self._cached_tracking_df: pd.DataFrame | None = None
        self._cached_summary: ActionsDetectorSummary | None = None
        self._materialized_dirty = True
        self._window_shifted = False
        self._person_slot_assignments_persistent: dict[str, str] = {}
        self._referee_slot_assignments_persistent: dict[str, str] = {}
        self._slot_last_xy: dict[str, tuple[float, float] | None] = {}
        self._slot_last_speed: dict[str, float] = {}
        self._slot_recent_observations: dict[str, deque[tuple[float, float]]] = {}

    @staticmethod
    def _clone_frame_map(frame_map: Any) -> dict[str, Any]:
        if not isinstance(frame_map, Mapping):
            return {}
        return {
            str(track_id): dict(payload) if isinstance(payload, Mapping) else payload
            for track_id, payload in frame_map.items()
        }

    @staticmethod
    def _clone_mapping(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            return {}
        return dict(payload)

    @staticmethod
    def _normalize_team_name(team_name: Any) -> str | None:
        if team_name is None:
            return None
        team_name = str(team_name).strip()
        return team_name or None

    @staticmethod
    def _normalize_role(role_name: Any) -> str | None:
        if role_name is None:
            return None
        role_name = str(role_name).strip()
        return role_name or None

    @staticmethod
    def _safe_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        arr = np.asarray(bbox[:4], dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            return None
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    @staticmethod
    def _safe_field_position(field_position: Any) -> tuple[float, float] | None:
        if not isinstance(field_position, (list, tuple)) or len(field_position) < 2:
            return None
        arr = np.asarray(field_position[:2], dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            return None
        return float(arr[0]), float(arr[1])

    @staticmethod
    def _should_use_payload_for_slot_tracking(payload: Mapping[str, Any]) -> bool:
        if not isinstance(payload, Mapping):
            return False
        if bool(payload.get("synthetic_seed")):
            return False
        bbox = payload.get("bbox")
        confidence = payload.get("confidence")
        if bbox is None and confidence is not None:
            try:
                if float(confidence) <= 0.0:
                    return False
            except (TypeError, ValueError):
                pass
        return True

    def update(self, frame_index: int, clean_packet: Mapping[str, Any]) -> ActionsDetectorSnapshot:
        expected_next = self._last_frame_index + 1
        if self._frames_seen > 0 and int(frame_index) != expected_next:
            raise ValueError(
                f"ActionsDetector esperaba frame_index={expected_next}, recibido {frame_index}"
            )

        tracks_frame = clean_packet.get("tracks_frame", {}) if isinstance(clean_packet, Mapping) else {}
        for class_name in TRACK_CLASSES:
            frame_map = tracks_frame.get(class_name, {}) if isinstance(tracks_frame, Mapping) else {}
            cloned_map = self._clone_frame_map(frame_map)
            self._snapshot[class_name].append(cloned_map)
            self._append_frame_observations(class_name, int(frame_index), cloned_map)

        possession_payload = clean_packet.get("possession", {}) if isinstance(clean_packet, Mapping) else {}
        self._snapshot["possession"].append(self._clone_mapping(possession_payload))

        self._frames_seen += 1
        self._last_frame_index = int(frame_index)
        self._trim_window_if_needed()
        self._release_stale_assignments()
        self._causal_update_materialized_state()
        self._materialized_dirty = True
        return self.snapshot_stats()

    def _append_frame_observations(
        self,
        class_name: str,
        frame_index: int,
        frame_map: Mapping[str, Any],
    ) -> None:
        if class_name not in self._records:
            return
        for raw_id, payload in frame_map.items():
            if not isinstance(payload, Mapping):
                continue
            if class_name in {"player", "referee"} and not self._should_use_payload_for_slot_tracking(payload):
                continue
            record = self._records[class_name].setdefault(
                str(raw_id),
                TrackRecord(raw_track_id=str(raw_id), class_name=class_name),
            )
            record.observations.append(
                TrackObservation(
                    frame_id=frame_index,
                    class_name=class_name,
                    team_name=self._normalize_team_name(payload.get("team")),
                    predicted_role=self._normalize_role(payload.get("predicted_role") or payload.get("predicted_role_frame")),
                    bbox=self._safe_bbox(payload.get("bbox")),
                    field_position_m=self._safe_field_position(payload.get("field_position_m")),
                )
            )

    def _trim_window_if_needed(self) -> None:
        max_frames = self.config.window_size_frames
        if max_frames is None or max_frames <= 0:
            return
        while len(self._snapshot["player"]) > max_frames:
            removed_frame_id = self._frames_seen - len(self._snapshot["player"])
            for class_name in self._snapshot.keys():
                self._snapshot[class_name].pop(0)
            self._prune_old_observations(removed_frame_id)
            self._window_shifted = True

    def _prune_old_observations(self, min_frame_id: int) -> None:
        for class_name, class_records in self._records.items():
            empty_ids: list[str] = []
            for raw_id, record in class_records.items():
                record.observations = [obs for obs in record.observations if obs.frame_id > min_frame_id]
                if not record.observations:
                    empty_ids.append(raw_id)
            for raw_id in empty_ids:
                del class_records[raw_id]

    def snapshot_stats(self) -> ActionsDetectorSnapshot:
        track_counts_by_class = {
            class_name: len(self._records[class_name])
            for class_name in TRACK_CLASSES
        }
        frame_count = len(self._snapshot["player"])
        window_start = 0 if frame_count <= 0 else int(self._snapshot_window_start())
        window_end = -1 if frame_count <= 0 else int(self._snapshot_window_end())
        return ActionsDetectorSnapshot(
            frame_count=frame_count,
            track_counts_by_class=track_counts_by_class,
            possession_frames=len(self._snapshot["possession"]),
            window_start_frame=window_start,
            window_end_frame=window_end,
        )

    def _snapshot_window_start(self) -> int:
        return self._last_frame_index - len(self._snapshot["player"]) + 1

    def _snapshot_window_end(self) -> int:
        return self._last_frame_index

    def build_tracks_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {
            class_name: [
                {
                    str(track_id): dict(payload) if isinstance(payload, Mapping) else payload
                    for track_id, payload in frame.items()
                }
                for frame in frames
            ]
            for class_name, frames in self._snapshot.items()
        }

    def build_tracking_dataframe(self) -> tuple[pd.DataFrame, ActionsDetectorSummary]:
        materialized_state, summary = self.build_materialized_slot_state()
        if self._cached_tracking_df is None or self._materialized_dirty:
            self._cached_tracking_df = materialized_state.tracking_df.copy()
            self._cached_summary = summary
            self._materialized_dirty = False
        return self._cached_tracking_df.copy(), self._cached_summary

    def build_materialized_slot_state(self) -> tuple[MaterializedSlotState, ActionsDetectorSummary]:
        frame_count = len(self._snapshot["player"])
        if frame_count <= 0:
            raise ValueError("No hay frames acumulados para construir el estado de PathCRF")
        if self._materialized_state is None:
            self._causal_update_materialized_state()
        assert self._materialized_state is not None
        synthetic_slots = self._current_synthetic_slots()
        self._materialized_state.synthetic_home_slots = list(synthetic_slots["home"])
        self._materialized_state.synthetic_away_slots = list(synthetic_slots["away"])
        self._materialized_state.synthetic_referee_slots = list(synthetic_slots["referee"])
        self._materialized_state.frame_count = frame_count
        summary = ActionsDetectorSummary(
            track_count_by_class={key: len(value) for key, value in self._records.items()},
            person_slot_assignments=dict(self._person_slot_assignments_persistent),
            referee_slot_assignments=dict(self._referee_slot_assignments_persistent),
            synthetic_home_slots=list(self._materialized_state.synthetic_home_slots),
            synthetic_away_slots=list(self._materialized_state.synthetic_away_slots),
            synthetic_referee_slots=list(self._materialized_state.synthetic_referee_slots),
            frames=frame_count,
        )
        self._cached_summary = summary
        return self._materialized_state, summary

    def _causal_update_materialized_state(self) -> None:
        if self._materialized_state is None:
            self._materialized_state = self._initialize_causal_state()
        current_len = len(self._snapshot["player"])
        built_len = int(self._materialized_state.frame_count)
        while built_len < current_len:
            local_frame_id = built_len
            self._append_causal_frame(local_frame_id)
            built_len += 1
        while self._materialized_state.frame_count > current_len:
            self._drop_oldest_materialized_row()
        self._cached_tracking_df = self._materialized_state.tracking_df.copy()
        self._window_shifted = False

    def _initialize_causal_state(self) -> MaterializedSlotState:
        person_slots = all_person_slots(self.config.expected_players_per_team)
        referee_slots = all_referee_slots(self.config.expected_referees)
        slot_tracks = {
            slot_name: pd.DataFrame(columns=["x", "y"], dtype=np.float32)
            for slot_name in person_slots
        }
        referee_tracks = {
            slot_name: pd.DataFrame(columns=["x", "y"], dtype=np.float32)
            for slot_name in referee_slots
        }
        ball_track = pd.DataFrame(columns=["x", "y"], dtype=np.float32)
        tracking_df = pd.DataFrame()
        for slot_name in [*person_slots, *referee_slots, "ball"]:
            self._slot_last_xy[slot_name] = None
            self._slot_last_speed[slot_name] = 0.0
        return MaterializedSlotState(
            frame_count=0,
            person_slot_assignments={},
            referee_slot_assignments={},
            slot_tracks=slot_tracks,
            referee_tracks=referee_tracks,
            ball_track=ball_track,
            carrier_series=pd.Series(dtype=object),
            owning_team_series=pd.Series(dtype=object),
            synthetic_home_slots=[f"home_{idx}" for idx in range(1, self.config.expected_players_per_team + 1)],
            synthetic_away_slots=[f"away_{idx}" for idx in range(1, self.config.expected_players_per_team + 1)],
            synthetic_referee_slots=all_referee_slots(self.config.expected_referees),
            tracking_df=tracking_df,
        )

    def _append_causal_frame(self, local_frame_id: int) -> None:
        assert self._materialized_state is not None
        person_positions = self._resolve_person_slot_positions_for_frame(local_frame_id)
        referee_positions = self._resolve_referee_slot_positions_for_frame(local_frame_id)
        self._append_xy_row_map(self._materialized_state.slot_tracks, person_positions)
        self._append_xy_row_map(self._materialized_state.referee_tracks, referee_positions)
        ball_xy, carrier_slot, owning_team = self._resolve_ball_row_for_frame(local_frame_id, person_positions)
        self._materialized_state.ball_track.loc[len(self._materialized_state.ball_track)] = [ball_xy[0], ball_xy[1]]

        row = self._build_tracking_row(
            local_frame_id=len(self._materialized_state.tracking_df),
            person_positions=person_positions,
            referee_positions=referee_positions,
            ball_xy=ball_xy,
            carrier_slot=carrier_slot,
            owning_team=owning_team,
        )
        self._materialized_state.tracking_df = pd.concat(
            [self._materialized_state.tracking_df, pd.DataFrame([row])],
            ignore_index=True,
        )
        self._materialized_state.carrier_series = pd.concat(
            [self._materialized_state.carrier_series, pd.Series([carrier_slot], dtype=object)],
            ignore_index=True,
        )
        self._materialized_state.owning_team_series = pd.concat(
            [self._materialized_state.owning_team_series, pd.Series([owning_team], dtype=object)],
            ignore_index=True,
        )
        self._materialized_state.frame_count += 1
        self._materialized_state.person_slot_assignments = dict(self._person_slot_assignments_persistent)
        self._materialized_state.referee_slot_assignments = dict(self._referee_slot_assignments_persistent)
        synthetic_slots = self._current_synthetic_slots()
        self._materialized_state.synthetic_home_slots = list(synthetic_slots["home"])
        self._materialized_state.synthetic_away_slots = list(synthetic_slots["away"])
        self._materialized_state.synthetic_referee_slots = list(synthetic_slots["referee"])

    def _drop_oldest_materialized_row(self) -> None:
        assert self._materialized_state is not None
        for slot_df in self._materialized_state.slot_tracks.values():
            if not slot_df.empty:
                slot_df.drop(index=slot_df.index[0], inplace=True)
                slot_df.reset_index(drop=True, inplace=True)
        for slot_df in self._materialized_state.referee_tracks.values():
            if not slot_df.empty:
                slot_df.drop(index=slot_df.index[0], inplace=True)
                slot_df.reset_index(drop=True, inplace=True)
        if not self._materialized_state.ball_track.empty:
            self._materialized_state.ball_track.drop(index=self._materialized_state.ball_track.index[0], inplace=True)
            self._materialized_state.ball_track.reset_index(drop=True, inplace=True)
        if not self._materialized_state.carrier_series.empty:
            self._materialized_state.carrier_series = self._materialized_state.carrier_series.iloc[1:].reset_index(drop=True)
        if not self._materialized_state.owning_team_series.empty:
            self._materialized_state.owning_team_series = self._materialized_state.owning_team_series.iloc[1:].reset_index(drop=True)
        if not self._materialized_state.tracking_df.empty:
            self._materialized_state.tracking_df = self._materialized_state.tracking_df.iloc[1:].reset_index(drop=True)
            self._materialized_state.tracking_df["frame_id"] = np.arange(len(self._materialized_state.tracking_df), dtype=np.int32)
            self._materialized_state.tracking_df["timestamp"] = (
                np.arange(len(self._materialized_state.tracking_df), dtype=np.float32) / float(self.config.fps)
            )
        self._materialized_state.frame_count = max(0, self._materialized_state.frame_count - 1)

    def _release_stale_assignments(self) -> None:
        self._person_slot_assignments_persistent = {
            track_id: slot_name
            for track_id in sorted(
                set(self._records["player"].keys()) | set(self._records["goalkeeper"].keys()),
                key=lambda item: int(item),
            )
            for slot_name in [canonical_id_to_person_slot(track_id)]
            if slot_name is not None
        }
        self._referee_slot_assignments_persistent = {
            track_id: slot_name
            for track_id in sorted(self._records["referee"].keys(), key=lambda item: int(item))
            for slot_name in [canonical_id_to_referee_slot(track_id)]
            if slot_name is not None
        }

    def _current_synthetic_slots(self) -> dict[str, list[str]]:
        used_person_slots = set(self._person_slot_assignments_persistent.values())
        used_ref_slots = set(self._referee_slot_assignments_persistent.values())
        return {
            "home": [
                f"home_{idx}"
                for idx in range(1, self.config.expected_players_per_team + 1)
                if f"home_{idx}" not in used_person_slots
            ],
            "away": [
                f"away_{idx}"
                for idx in range(1, self.config.expected_players_per_team + 1)
                if f"away_{idx}" not in used_person_slots
            ],
            "referee": [
                f"referee_{idx}"
                for idx in range(1, self.config.expected_referees + 1)
                if f"referee_{idx}" not in used_ref_slots
            ],
        }

    def _resolve_person_slot_positions_for_frame(self, local_frame_id: int) -> dict[str, tuple[float, float]]:
        frame_player = self._snapshot["player"][local_frame_id] if local_frame_id < len(self._snapshot["player"]) else {}
        frame_goalkeeper = self._snapshot["goalkeeper"][local_frame_id] if local_frame_id < len(self._snapshot["goalkeeper"]) else {}
        positions: dict[str, tuple[float, float]] = {}
        for slot_name in all_person_slots(self.config.expected_players_per_team):
            canonical_id = person_slot_to_canonical_id(slot_name)
            payload = None
            if canonical_id is not None:
                payload = frame_goalkeeper.get(str(canonical_id))
                if payload is None:
                    payload = frame_player.get(str(canonical_id))
            observed_xy = self._person_observed_xy_from_payload(payload, slot_name)
            positions[slot_name] = self._compute_causal_slot_xy(
                slot_name=slot_name,
                observed_xy=observed_xy,
                fallback_xy=self._fallback_slot_xy(slot_name),
                max_speed_mps=self.config.player_outlier_speed_mps,
            )
        return positions

    def _person_observed_xy_from_payload(
        self,
        payload: Any,
        slot_name: str,
    ) -> tuple[float, float] | None:
        if not isinstance(payload, Mapping):
            return None
        if not slot_name.endswith("_1") and not self._should_use_payload_for_slot_tracking(payload):
            return None
        return self._safe_field_position(payload.get("field_position_m"))

    def _resolve_referee_slot_positions_for_frame(self, local_frame_id: int) -> dict[str, tuple[float, float]]:
        frame_referee = self._snapshot["referee"][local_frame_id] if local_frame_id < len(self._snapshot["referee"]) else {}
        positions: dict[str, tuple[float, float]] = {}
        for slot_name in all_referee_slots(self.config.expected_referees):
            canonical_id = referee_slot_to_canonical_id(slot_name)
            payload = frame_referee.get(str(canonical_id)) if canonical_id is not None else None
            observed_xy = None
            if isinstance(payload, Mapping) and self._should_use_payload_for_slot_tracking(payload):
                observed_xy = self._safe_field_position(payload.get("field_position_m"))
            positions[slot_name] = self._compute_causal_slot_xy(
                slot_name=slot_name,
                observed_xy=observed_xy,
                fallback_xy=self._fallback_slot_xy(slot_name),
                max_speed_mps=self.config.referee_outlier_speed_mps,
            )
        return positions

    def _resolve_ball_row_for_frame(
        self,
        local_frame_id: int,
        person_positions: Mapping[str, tuple[float, float]],
    ) -> tuple[tuple[float, float], str | None, str | None]:
        del local_frame_id, person_positions
        ball_xy = self._compute_causal_slot_xy(
            slot_name="ball",
            observed_xy=None,
            fallback_xy=(self.config.pitch_length_m / 2.0, self.config.pitch_width_m / 2.0),
            max_speed_mps=self.config.ball_outlier_speed_mps,
        )
        return ball_xy, None, None

    def _compute_causal_slot_xy(
        self,
        *,
        slot_name: str,
        observed_xy: tuple[float, float] | None,
        fallback_xy: tuple[float, float],
        max_speed_mps: float,
    ) -> tuple[float, float]:
        prev_xy = self._slot_last_xy.get(slot_name)
        history = self._slot_recent_observations.setdefault(
            slot_name,
            deque(maxlen=max(3, int(self.config.smoothing_window))),
        )
        if observed_xy is not None:
            history.append((float(observed_xy[0]), float(observed_xy[1])))
        if history:
            observed_arr = np.asarray(list(history), dtype=np.float32)
            median_xy = np.median(observed_arr, axis=0)
            source_xy = (float(median_xy[0]), float(median_xy[1]))
        else:
            source_xy = observed_xy if observed_xy is not None else (prev_xy if prev_xy is not None else fallback_xy)
        source_arr = np.asarray(source_xy, dtype=np.float32)
        if prev_xy is None:
            smoothed_arr = source_arr
        else:
            prev_arr = np.asarray(prev_xy, dtype=np.float32)
            blended = (0.45 * source_arr) + (0.55 * prev_arr)
            delta = blended - prev_arr
            dt = 1.0 / float(self.config.fps)
            max_step = float(max_speed_mps) * dt
            step = float(np.linalg.norm(delta))
            if step > max_step > 1e-6:
                blended = prev_arr + (delta * (max_step / step))
            smoothed_arr = blended
        smoothed_arr[0] = float(np.clip(smoothed_arr[0], 0.0, self.config.pitch_length_m))
        smoothed_arr[1] = float(np.clip(smoothed_arr[1], 0.0, self.config.pitch_width_m))
        self._slot_last_xy[slot_name] = (float(smoothed_arr[0]), float(smoothed_arr[1]))
        return self._slot_last_xy[slot_name]

    def _fallback_slot_xy(self, slot_name: str) -> tuple[float, float]:
        if slot_name.startswith("home_"):
            return tuple(self._team_template_for_side("home")[int(slot_name.split("_")[-1]) - 1].tolist())
        if slot_name.startswith("away_"):
            return tuple(self._team_template_for_side("away")[int(slot_name.split("_")[-1]) - 1].tolist())
        if slot_name.startswith("referee_"):
            return tuple(REFEREE_TEMPLATE[int(slot_name.split("_")[-1]) - 1].tolist())
        return (self.config.pitch_length_m / 2.0, self.config.pitch_width_m / 2.0)

    def _append_xy_row_map(
        self,
        slot_tracks: Mapping[str, pd.DataFrame],
        positions: Mapping[str, tuple[float, float]],
    ) -> None:
        for slot_name, slot_df in slot_tracks.items():
            xy = positions.get(slot_name, self._fallback_slot_xy(slot_name))
            slot_df.loc[len(slot_df)] = [float(xy[0]), float(xy[1])]

    def _build_tracking_row(
        self,
        *,
        local_frame_id: int,
        person_positions: Mapping[str, tuple[float, float]],
        referee_positions: Mapping[str, tuple[float, float]],
        ball_xy: tuple[float, float],
        carrier_slot: str | None,
        owning_team: str | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "frame_id": int(local_frame_id),
            "period_id": 1,
            "timestamp": float(local_frame_id) / float(self.config.fps),
            "phase_id": 1,
            "episode_id": 1,
            "ball_state": "alive",
            "ball_owning_team_id": owning_team,
            "player_id": carrier_slot,
        }
        for slot_name in all_person_slots(self.config.expected_players_per_team):
            self._append_motion_features_to_row(row, slot_name, person_positions[slot_name])
        for slot_name in all_referee_slots(self.config.expected_referees):
            self._append_motion_features_to_row(row, slot_name, referee_positions[slot_name])
        self._append_motion_features_to_row(row, "ball", ball_xy)
        return row

    def _append_motion_features_to_row(
        self,
        row: dict[str, Any],
        slot_name: str,
        xy: tuple[float, float],
    ) -> None:
        prefix = f"{slot_name}_"
        prev_xy = None
        if self._materialized_state is not None and len(self._materialized_state.tracking_df) > 0:
            prev_x = self._materialized_state.tracking_df.iloc[-1].get(f"{slot_name}_x")
            prev_y = self._materialized_state.tracking_df.iloc[-1].get(f"{slot_name}_y")
            if pd.notna(prev_x) and pd.notna(prev_y):
                prev_xy = (float(prev_x), float(prev_y))
        dt = 1.0 / float(self.config.fps)
        if prev_xy is None:
            vx = 0.0
            vy = 0.0
            speed = 0.0
            accel = 0.0
        else:
            vx = (float(xy[0]) - float(prev_xy[0])) / dt
            vy = (float(xy[1]) - float(prev_xy[1])) / dt
            speed = float(np.sqrt((vx ** 2) + (vy ** 2)))
            accel = (speed - float(self._slot_last_speed.get(slot_name, 0.0))) / dt
        self._slot_last_speed[slot_name] = float(speed)
        row[f"{prefix}x"] = float(xy[0])
        row[f"{prefix}y"] = float(xy[1])
        row[f"{prefix}vx"] = float(vx)
        row[f"{prefix}vy"] = float(vy)
        row[f"{prefix}speed"] = float(speed)
        row[f"{prefix}accel"] = float(accel)

    def _team_template_for_side(self, side: str) -> np.ndarray:
        if str(side).strip().lower() == "away":
            mirrored = HOME_TEAM_TEMPLATE.copy()
            mirrored[:, 0] = self.config.pitch_length_m - mirrored[:, 0]
            return mirrored
        return HOME_TEAM_TEMPLATE
