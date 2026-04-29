from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter


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
    framewise_person_slot_assignment: bool = False
    player_outlier_speed_mps: float = 14.0
    referee_outlier_speed_mps: float = 12.0
    ball_outlier_speed_mps: float = 35.0
    smoothing_window: int = 9
    smoothing_center: bool = True
    savgol_window: int = 11
    savgol_polyorder: int = 2
    smoothing_passes: int = 2
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
            if class_name in {"player", "goalkeeper", "referee"} and not self._should_use_payload_for_slot_tracking(payload):
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
        records = self._build_window_records()
        synthetic_slots = self._current_synthetic_slots()
        self._materialized_state.synthetic_home_slots = list(synthetic_slots["home"])
        self._materialized_state.synthetic_away_slots = list(synthetic_slots["away"])
        self._materialized_state.synthetic_referee_slots = list(synthetic_slots["referee"])
        self._materialized_state.frame_count = frame_count
        summary = ActionsDetectorSummary(
            track_count_by_class={key: len(value) for key, value in records.items()},
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
        person_slots = [f"home_{idx}" for idx in range(1, 12)] + [f"away_{idx}" for idx in range(1, 12)]
        referee_slots = [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)]
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
            synthetic_referee_slots=[f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)],
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
        active_person_ids = set(self._records["player"].keys()) | set(self._records["goalkeeper"].keys())
        active_ref_ids = set(self._records["referee"].keys())
        self._person_slot_assignments_persistent = {
            track_id: slot_name
            for track_id, slot_name in self._person_slot_assignments_persistent.items()
            if track_id in active_person_ids
        }
        self._referee_slot_assignments_persistent = {
            track_id: slot_name
            for track_id, slot_name in self._referee_slot_assignments_persistent.items()
            if track_id in active_ref_ids
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
        detections_by_side = {"home": [], "away": []}
        for class_name, frame_map in (("player", frame_player), ("goalkeeper", frame_goalkeeper)):
            for raw_id, payload in frame_map.items():
                if not isinstance(payload, Mapping) or not self._should_use_payload_for_slot_tracking(payload):
                    continue
                xy = self._safe_field_position(payload.get("field_position_m"))
                if xy is None:
                    continue
                side = self._resolve_payload_side(payload, xy, {})
                detections_by_side[side].append(
                    {
                        "raw_id": str(raw_id),
                        "class_name": class_name,
                        "role": self._normalize_role(payload.get("predicted_role") or payload.get("predicted_role_frame")),
                        "xy": xy,
                    }
                )

        self._assign_new_person_slots(detections_by_side)
        positions: dict[str, tuple[float, float]] = {}
        for track_id, slot_name in self._person_slot_assignments_persistent.items():
            payload = frame_player.get(track_id)
            if payload is None:
                payload = frame_goalkeeper.get(track_id)
            observed_xy = self._safe_field_position(payload.get("field_position_m")) if isinstance(payload, Mapping) else None
            positions[slot_name] = self._compute_causal_slot_xy(
                slot_name=slot_name,
                observed_xy=observed_xy,
                fallback_xy=self._fallback_slot_xy(slot_name),
                max_speed_mps=self.config.player_outlier_speed_mps,
            )

        for slot_name in [f"home_{idx}" for idx in range(1, 12)] + [f"away_{idx}" for idx in range(1, 12)]:
            positions.setdefault(
                slot_name,
                self._compute_causal_slot_xy(
                    slot_name=slot_name,
                    observed_xy=None,
                    fallback_xy=self._fallback_slot_xy(slot_name),
                    max_speed_mps=self.config.player_outlier_speed_mps,
                ),
            )
        return positions

    def _assign_new_person_slots(self, detections_by_side: Mapping[str, list[dict[str, Any]]]) -> None:
        for side in ("home", "away"):
            side_detections = detections_by_side.get(side, [])
            new_detections = [det for det in side_detections if det["raw_id"] not in self._person_slot_assignments_persistent]
            if not new_detections:
                continue
            free_slots = [
                f"{side}_{idx}"
                for idx in range(1, self.config.expected_players_per_team + 1)
                if f"{side}_{idx}" not in self._person_slot_assignments_persistent.values()
            ]
            if not free_slots:
                continue
            goalkeeper = self._select_goalkeeper_detection(side, new_detections)
            if goalkeeper is not None and f"{side}_1" in free_slots:
                self._person_slot_assignments_persistent[goalkeeper["raw_id"]] = f"{side}_1"
                free_slots.remove(f"{side}_1")
                new_detections = [det for det in new_detections if det["raw_id"] != goalkeeper["raw_id"]]
            if not new_detections or not free_slots:
                continue
            template = self._team_template_for_side(side)
            cost_matrix = np.zeros((len(new_detections), len(free_slots)), dtype=np.float32)
            for row_idx, detection in enumerate(new_detections):
                detection_xy = np.asarray(detection["xy"], dtype=np.float32)
                for col_idx, slot_name in enumerate(free_slots):
                    slot_idx = int(slot_name.split("_")[-1]) - 1
                    template_xy = template[slot_idx]
                    cost_matrix[row_idx, col_idx] = float(np.linalg.norm(detection_xy - template_xy))
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
                self._person_slot_assignments_persistent[new_detections[row_idx]["raw_id"]] = free_slots[col_idx]

    def _resolve_referee_slot_positions_for_frame(self, local_frame_id: int) -> dict[str, tuple[float, float]]:
        frame_referee = self._snapshot["referee"][local_frame_id] if local_frame_id < len(self._snapshot["referee"]) else {}
        detections = []
        for raw_id, payload in frame_referee.items():
            if not isinstance(payload, Mapping) or not self._should_use_payload_for_slot_tracking(payload):
                continue
            xy = self._safe_field_position(payload.get("field_position_m"))
            if xy is None:
                continue
            detections.append({"raw_id": str(raw_id), "xy": xy})
        self._assign_new_referee_slots(detections)
        positions: dict[str, tuple[float, float]] = {}
        for track_id, slot_name in self._referee_slot_assignments_persistent.items():
            payload = frame_referee.get(track_id)
            observed_xy = self._safe_field_position(payload.get("field_position_m")) if isinstance(payload, Mapping) else None
            positions[slot_name] = self._compute_causal_slot_xy(
                slot_name=slot_name,
                observed_xy=observed_xy,
                fallback_xy=self._fallback_slot_xy(slot_name),
                max_speed_mps=self.config.referee_outlier_speed_mps,
            )
        for slot_name in [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)]:
            positions.setdefault(
                slot_name,
                self._compute_causal_slot_xy(
                    slot_name=slot_name,
                    observed_xy=None,
                    fallback_xy=self._fallback_slot_xy(slot_name),
                    max_speed_mps=self.config.referee_outlier_speed_mps,
                ),
            )
        return positions

    def _assign_new_referee_slots(self, detections: list[dict[str, Any]]) -> None:
        new_detections = [det for det in detections if det["raw_id"] not in self._referee_slot_assignments_persistent]
        free_slots = [
            f"referee_{idx}"
            for idx in range(1, self.config.expected_referees + 1)
            if f"referee_{idx}" not in self._referee_slot_assignments_persistent.values()
        ]
        if not new_detections or not free_slots:
            return
        ordered_detections = sorted(new_detections, key=lambda det: (det["xy"][0], det["xy"][1], det["raw_id"]))
        for slot_name, det in zip(free_slots, ordered_detections):
            self._referee_slot_assignments_persistent[det["raw_id"]] = slot_name

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
        source_xy = observed_xy if observed_xy is not None else (prev_xy if prev_xy is not None else fallback_xy)
        source_arr = np.asarray(source_xy, dtype=np.float32)
        if prev_xy is None:
            smoothed_arr = source_arr
        else:
            prev_arr = np.asarray(prev_xy, dtype=np.float32)
            blended = (0.35 * source_arr) + (0.65 * prev_arr)
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
        for slot_name in [f"home_{idx}" for idx in range(1, 12)] + [f"away_{idx}" for idx in range(1, 12)]:
            self._append_motion_features_to_row(row, slot_name, person_positions[slot_name])
        for slot_name in [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)]:
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

    def _must_rebuild_materialized_state(
        self,
        *,
        frame_count: int,
        person_slot_assignments: Mapping[str, str],
        referee_slot_assignments: Mapping[str, str],
    ) -> bool:
        state = self._materialized_state
        if state is None:
            return True
        if self._window_shifted:
            return True
        if state.person_slot_assignments != dict(person_slot_assignments):
            return True
        if state.referee_slot_assignments != dict(referee_slot_assignments):
            return True
        if state.frame_count != max(frame_count - 1, 0):
            return True
        return False

    def _rebuild_materialized_state(
        self,
        *,
        records: dict[str, dict[str, TrackRecord]],
        person_assignments: Mapping[str, str],
        referee_assignments: Mapping[str, str],
        synthetic_slots: Mapping[str, list[str]],
    ) -> MaterializedSlotState:
        snapshot = self.build_tracks_snapshot()
        frame_count = len(snapshot["player"])
        raw_to_slot = {raw_id: slot for slot, raw_id in person_assignments.items()}
        raw_ref_to_slot = {raw_id: slot for slot, raw_id in referee_assignments.items()}
        all_person_slots = [f"home_{idx}" for idx in range(1, 12)] + [f"away_{idx}" for idx in range(1, 12)]
        all_ref_slots = [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)]
        _, side_by_team = self._group_person_records_by_side([*records["player"].values(), *records["goalkeeper"].values()])

        if self.config.framewise_person_slot_assignment:
            slot_tracks = self._build_person_slot_tracks(
                tracks=snapshot,
                frame_count=frame_count,
                side_by_team=side_by_team,
                ordered_slots=all_person_slots,
            )
        else:
            slot_tracks = self._build_slot_track_arrays(snapshot, frame_count, raw_to_slot, all_person_slots)

        referee_tracks = self._build_slot_track_arrays(
            snapshot,
            frame_count,
            raw_ref_to_slot,
            all_ref_slots,
            class_names=("referee",),
            max_speed_mps=self.config.referee_outlier_speed_mps,
        )
        self._fill_team_templates(slot_tracks, "home")
        self._fill_team_templates(slot_tracks, "away")
        self._fill_referee_templates(referee_tracks)

        ball_df, carrier_series, owning_team_series = self._build_ball_and_carrier_series(
            tracks=snapshot,
            frame_count=frame_count,
            raw_to_slot=raw_to_slot,
            slot_tracks=slot_tracks,
        )
        if not self.config.export_possession_targets:
            carrier_series = pd.Series([None] * frame_count, dtype=object)
            owning_team_series = pd.Series([None] * frame_count, dtype=object)

        self._window_shifted = False
        self._cached_tracking_df = None
        return MaterializedSlotState(
            frame_count=frame_count,
            person_slot_assignments={raw_id: slot for raw_id, slot in raw_to_slot.items()},
            referee_slot_assignments={raw_id: slot for raw_id, slot in raw_ref_to_slot.items()},
            slot_tracks=slot_tracks,
            referee_tracks=referee_tracks,
            ball_track=ball_df,
            carrier_series=carrier_series,
            owning_team_series=owning_team_series,
            synthetic_home_slots=list(synthetic_slots["home"]),
            synthetic_away_slots=list(synthetic_slots["away"]),
            synthetic_referee_slots=list(synthetic_slots["referee"]),
        )

    def _append_materialized_frame(
        self,
        *,
        person_slot_assignments: Mapping[str, str],
        referee_slot_assignments: Mapping[str, str],
    ) -> None:
        state = self._materialized_state
        if state is None:
            return
        new_index = state.frame_count

        self._append_slot_frame(
            slot_tracks=state.slot_tracks,
            slot_assignments=person_slot_assignments,
            row_index=new_index,
            frame_map=self._snapshot["player"][-1],
            allow_payload_filter=True,
        )
        self._append_slot_frame(
            slot_tracks=state.slot_tracks,
            slot_assignments=person_slot_assignments,
            row_index=new_index,
            frame_map=self._snapshot["goalkeeper"][-1],
            allow_payload_filter=True,
        )
        self._append_slot_frame(
            slot_tracks=state.referee_tracks,
            slot_assignments=referee_slot_assignments,
            row_index=new_index,
            frame_map=self._snapshot["referee"][-1],
            allow_payload_filter=True,
        )

        for slot_df in state.slot_tracks.values():
            if new_index not in slot_df.index:
                slot_df.loc[new_index, ["x", "y"]] = np.nan
        for slot_df in state.referee_tracks.values():
            if new_index not in slot_df.index:
                slot_df.loc[new_index, ["x", "y"]] = np.nan

        state.frame_count += 1
        for slot_name, slot_df in list(state.slot_tracks.items()):
            state.slot_tracks[slot_name] = self._stabilize_xy(
                self._interpolate_xy(slot_df),
                max_speed_mps=self.config.player_outlier_speed_mps,
            )
        for slot_name, slot_df in list(state.referee_tracks.items()):
            state.referee_tracks[slot_name] = self._stabilize_xy(
                self._interpolate_xy(slot_df),
                max_speed_mps=self.config.referee_outlier_speed_mps,
            )

        self._fill_team_templates(state.slot_tracks, "home")
        self._fill_team_templates(state.slot_tracks, "away")
        self._fill_referee_templates(state.referee_tracks)

        state.ball_track, state.carrier_series, state.owning_team_series = self._build_ball_and_carrier_series(
            tracks=self.build_tracks_snapshot(),
            frame_count=state.frame_count,
            raw_to_slot=person_slot_assignments,
            slot_tracks=state.slot_tracks,
        )
        if not self.config.export_possession_targets:
            state.carrier_series = pd.Series([None] * state.frame_count, dtype=object)
            state.owning_team_series = pd.Series([None] * state.frame_count, dtype=object)

    def _append_slot_frame(
        self,
        *,
        slot_tracks: Mapping[str, pd.DataFrame],
        slot_assignments: Mapping[str, str],
        row_index: int,
        frame_map: Mapping[str, Any],
        allow_payload_filter: bool,
    ) -> None:
        for raw_id, payload in frame_map.items():
            slot_name = slot_assignments.get(str(raw_id))
            if slot_name is None or slot_name not in slot_tracks:
                continue
            if allow_payload_filter and not self._should_use_payload_for_slot_tracking(payload):
                continue
            field_position = self._safe_field_position(payload.get("field_position_m"))
            if field_position is None:
                continue
            slot_tracks[slot_name].loc[row_index, ["x", "y"]] = field_position

    def _build_tracking_dataframe_from_materialized_state(self, materialized_state: MaterializedSlotState) -> pd.DataFrame:
        frame_count = materialized_state.frame_count
        state_df = pd.DataFrame(
            {
                "frame_id": np.arange(frame_count, dtype=np.int32),
                "period_id": np.ones(frame_count, dtype=np.int16),
                "timestamp": np.arange(frame_count, dtype=np.float32) / float(self.config.fps),
                "phase_id": np.ones(frame_count, dtype=np.int16),
                "episode_id": np.ones(frame_count, dtype=np.int16),
                "ball_state": np.full(frame_count, "alive", dtype=object),
                "ball_owning_team_id": materialized_state.owning_team_series,
                "player_id": materialized_state.carrier_series,
            }
        )
        output_df = state_df.copy()
        for slot_name, slot_df in materialized_state.slot_tracks.items():
            output_df = pd.concat([output_df, self._compute_motion_features(slot_df).add_prefix(f"{slot_name}_")], axis=1)
        for slot_name, slot_df in materialized_state.referee_tracks.items():
            output_df = pd.concat([output_df, self._compute_motion_features(slot_df).add_prefix(f"{slot_name}_")], axis=1)
        output_df = pd.concat([output_df, self._compute_motion_features(materialized_state.ball_track).add_prefix("ball_")], axis=1)
        return output_df

    def _build_window_records(self) -> dict[str, dict[str, TrackRecord]]:
        return {
            class_name: {
                raw_id: TrackRecord(
                    raw_track_id=record.raw_track_id,
                    class_name=record.class_name,
                    observations=list(record.observations),
                )
                for raw_id, record in class_records.items()
            }
            for class_name, class_records in self._records.items()
        }

    def _assign_slots(
        self,
        records: dict[str, dict[str, TrackRecord]],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
        person_records = list(records["player"].values()) + list(records["goalkeeper"].values())
        person_records = [record for record in person_records if record.observations]
        referee_records = [record for record in records["referee"].values() if record.observations]

        team_records, side_by_team = self._group_person_records_by_side(person_records)
        home_records = [record for team, recs in team_records.items() if side_by_team.get(team) == "home" for record in recs]
        away_records = [record for team, recs in team_records.items() if side_by_team.get(team) == "away" for record in recs]

        person_assignments: dict[str, str] = {}
        person_assignments.update(self._assign_team_slots("home", home_records))
        person_assignments.update(self._assign_team_slots("away", away_records))

        referee_assignments: dict[str, str] = {}
        for idx, record in enumerate(
            sorted(referee_records, key=lambda item: (item.first_frame, int(item.raw_track_id))),
            start=1,
        ):
            if idx > self.config.expected_referees:
                break
            referee_assignments[f"referee_{idx}"] = record.raw_track_id

        synthetic_slots = {
            "home": [
                f"home_{idx}"
                for idx in range(1, self.config.expected_players_per_team + 1)
                if f"home_{idx}" not in person_assignments
            ],
            "away": [
                f"away_{idx}"
                for idx in range(1, self.config.expected_players_per_team + 1)
                if f"away_{idx}" not in person_assignments
            ],
            "referee": [
                f"referee_{idx}"
                for idx in range(1, self.config.expected_referees + 1)
                if f"referee_{idx}" not in referee_assignments
            ],
        }
        return person_assignments, referee_assignments, synthetic_slots

    def _group_person_records_by_side(
        self,
        person_records: list[TrackRecord],
    ) -> tuple[dict[str, list[TrackRecord]], dict[str, str]]:
        team_records: dict[str, list[TrackRecord]] = {}
        for record in person_records:
            team_name = record.stable_team_name()
            if team_name:
                team_records.setdefault(team_name, []).append(record)

        ordered_teams = self._order_teams_by_side(team_records)
        side_by_team: dict[str, str] = {}
        if ordered_teams:
            side_by_team[ordered_teams[0]] = "home"
        if len(ordered_teams) > 1:
            side_by_team[ordered_teams[1]] = "away"

        unknown_records = [record for record in person_records if record.stable_team_name() is None]
        for record in unknown_records:
            median_x = record.median_x(self.config.pitch_length_m / 2.0)
            side = "home" if median_x <= (self.config.pitch_length_m / 2.0) else "away"
            synthetic_team_name = f"{side}_unknown"
            side_by_team[synthetic_team_name] = side
            team_records.setdefault(synthetic_team_name, []).append(record)
        return team_records, side_by_team

    def _order_teams_by_side(self, team_records: Mapping[str, list[TrackRecord]]) -> list[str]:
        def team_median_x(item: tuple[str, list[TrackRecord]]) -> float:
            _, records = item
            xs = [record.median_field_position()[0] for record in records if record.median_field_position() is not None]
            if not xs:
                return self.config.pitch_length_m / 2.0
            return float(np.median(xs))

        ordered = sorted(team_records.items(), key=team_median_x)
        return [team_name for team_name, _ in ordered[:2]]

    def _assign_team_slots(self, side: str, records: list[TrackRecord]) -> dict[str, str]:
        if not records:
            return {}
        assignments: dict[str, str] = {}
        goalkeeper = self._select_goalkeeper(side, records)
        if goalkeeper is not None:
            assignments[f"{side}_1"] = goalkeeper.raw_track_id

        remaining = [record for record in records if goalkeeper is None or record.raw_track_id != goalkeeper.raw_track_id]
        template = self._team_template_for_side(side)
        remaining_slots = [f"{side}_{idx}" for idx in range(2, self.config.expected_players_per_team + 1)]
        if remaining and remaining_slots:
            cost_matrix = np.zeros((len(remaining), len(remaining_slots)), dtype=np.float32)
            fallback_x = 0.0 if side == "home" else self.config.pitch_length_m
            for row_idx, record in enumerate(remaining):
                xy = record.median_field_position()
                if xy is None:
                    xy = (record.median_x(fallback_x), self.config.pitch_width_m / 2.0)
                record_xy = np.asarray(xy, dtype=np.float32)
                for col_idx, slot_name in enumerate(remaining_slots):
                    slot_idx = int(slot_name.split("_")[-1]) - 1
                    template_xy = template[slot_idx]
                    distance = float(np.linalg.norm(record_xy - template_xy))
                    role_penalty = 0.0
                    if record.class_name == "goalkeeper" or record.stable_role() == "POR":
                        role_penalty += 25.0
                    y_penalty = 0.15 * abs(float(record_xy[1] - template_xy[1]))
                    cost_matrix[row_idx, col_idx] = distance + y_penalty + role_penalty
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
                assignments[remaining_slots[col_idx]] = remaining[row_idx].raw_track_id
        return assignments

    def _select_goalkeeper(self, side: str, records: list[TrackRecord]) -> TrackRecord | None:
        candidates = [
            record
            for record in records
            if record.class_name == "goalkeeper" or record.stable_role() == "POR"
        ]
        if not candidates:
            return None
        reverse = side == "away"
        return sorted(
            candidates,
            key=lambda item: item.median_x(self.config.pitch_length_m / 2.0),
            reverse=reverse,
        )[0]

    def _build_tracking_dataframe_internal(
        self,
        tracks: Mapping[str, Any],
        frame_count: int,
        records: dict[str, dict[str, TrackRecord]],
        person_assignments: Mapping[str, str],
        referee_assignments: Mapping[str, str],
    ) -> pd.DataFrame:
        raw_to_slot = {raw_id: slot for slot, raw_id in person_assignments.items()}
        raw_ref_to_slot = {raw_id: slot for slot, raw_id in referee_assignments.items()}
        all_person_slots = [f"home_{idx}" for idx in range(1, 12)] + [f"away_{idx}" for idx in range(1, 12)]
        all_ref_slots = [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)]
        _, side_by_team = self._group_person_records_by_side([*records["player"].values(), *records["goalkeeper"].values()])

        if self.config.framewise_person_slot_assignment:
            slot_tracks = self._build_person_slot_tracks(
                tracks=tracks,
                frame_count=frame_count,
                side_by_team=side_by_team,
                ordered_slots=all_person_slots,
            )
        else:
            slot_tracks = self._build_slot_track_arrays(tracks, frame_count, raw_to_slot, all_person_slots)

        referee_tracks = self._build_slot_track_arrays(
            tracks,
            frame_count,
            raw_ref_to_slot,
            all_ref_slots,
            class_names=("referee",),
            max_speed_mps=self.config.referee_outlier_speed_mps,
        )
        self._fill_team_templates(slot_tracks, "home")
        self._fill_team_templates(slot_tracks, "away")
        self._fill_referee_templates(referee_tracks)

        ball_df, carrier_series, owning_team_series = self._build_ball_and_carrier_series(
            tracks=tracks,
            frame_count=frame_count,
            raw_to_slot=raw_to_slot,
            slot_tracks=slot_tracks,
        )

        if not self.config.export_possession_targets:
            carrier_series = pd.Series([None] * frame_count, dtype=object)
            owning_team_series = pd.Series([None] * frame_count, dtype=object)

        state_df = pd.DataFrame(
            {
                "frame_id": np.arange(frame_count, dtype=np.int32),
                "period_id": np.ones(frame_count, dtype=np.int16),
                "timestamp": np.arange(frame_count, dtype=np.float32) / float(self.config.fps),
                "phase_id": np.ones(frame_count, dtype=np.int16),
                "episode_id": np.ones(frame_count, dtype=np.int16),
                "ball_state": np.full(frame_count, "alive", dtype=object),
                "ball_owning_team_id": owning_team_series,
                "player_id": carrier_series,
            }
        )

        output_df = state_df.copy()
        for slot_name, slot_df in slot_tracks.items():
            output_df = pd.concat([output_df, self._compute_motion_features(slot_df).add_prefix(f"{slot_name}_")], axis=1)
        for slot_name, slot_df in referee_tracks.items():
            output_df = pd.concat([output_df, self._compute_motion_features(slot_df).add_prefix(f"{slot_name}_")], axis=1)
        output_df = pd.concat([output_df, self._compute_motion_features(ball_df).add_prefix("ball_")], axis=1)
        return output_df

    def _build_person_slot_tracks(
        self,
        tracks: Mapping[str, Any],
        frame_count: int,
        side_by_team: Mapping[str, str],
        ordered_slots: list[str],
    ) -> dict[str, pd.DataFrame]:
        slot_frames = {
            slot_name: pd.DataFrame(index=np.arange(frame_count), columns=["x", "y"], dtype=np.float32)
            for slot_name in ordered_slots
        }
        continuity_state = {
            "home": {
                "xy": {f"home_{idx}": None for idx in range(1, self.config.expected_players_per_team + 1)},
                "raw_id": {f"home_{idx}": None for idx in range(1, self.config.expected_players_per_team + 1)},
            },
            "away": {
                "xy": {f"away_{idx}": None for idx in range(1, self.config.expected_players_per_team + 1)},
                "raw_id": {f"away_{idx}": None for idx in range(1, self.config.expected_players_per_team + 1)},
            },
        }

        for frame_id in range(frame_count):
            detections_by_side = {"home": [], "away": []}
            for class_name in ("player", "goalkeeper"):
                frames = tracks.get(class_name, [])
                if frame_id >= len(frames):
                    continue
                frame_map = frames[frame_id]
                if not isinstance(frame_map, Mapping):
                    continue
                for raw_id, payload in frame_map.items():
                    if not isinstance(payload, Mapping):
                        continue
                    field_position = self._safe_field_position(payload.get("field_position_m"))
                    if field_position is None:
                        continue
                    side = self._resolve_payload_side(payload, field_position, side_by_team)
                    detections_by_side[side].append(
                        {
                            "raw_id": str(raw_id),
                            "class_name": class_name,
                            "role": self._normalize_role(payload.get("predicted_role") or payload.get("predicted_role_frame")),
                            "xy": field_position,
                        }
                    )

            for side in ("home", "away"):
                slot_names = [f"{side}_{idx}" for idx in range(1, self.config.expected_players_per_team + 1)]
                assigned = self._assign_frame_team_detections(
                    detections=detections_by_side[side],
                    side=side,
                    slot_names=slot_names,
                    state=continuity_state[side],
                )
                for slot_name, detection in assigned.items():
                    slot_frames[slot_name].loc[frame_id, ["x", "y"]] = detection["xy"]
                    continuity_state[side]["xy"][slot_name] = detection["xy"]
                    continuity_state[side]["raw_id"][slot_name] = detection["raw_id"]

        for slot_name, slot_df in slot_frames.items():
            slot_frames[slot_name] = self._stabilize_xy(
                self._interpolate_xy(slot_df),
                max_speed_mps=self.config.player_outlier_speed_mps,
            )
        return slot_frames

    def _resolve_payload_side(
        self,
        payload: Mapping[str, Any],
        field_position: tuple[float, float],
        side_by_team: Mapping[str, str],
    ) -> str:
        team_name = self._normalize_team_name(payload.get("team"))
        side = side_by_team.get(team_name) if team_name is not None else None
        if side in {"home", "away"}:
            return side
        return "home" if field_position[0] <= (self.config.pitch_length_m / 2.0) else "away"

    def _assign_frame_team_detections(
        self,
        detections: list[dict[str, Any]],
        side: str,
        slot_names: list[str],
        state: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        assigned: dict[str, dict[str, Any]] = {}
        if not detections:
            return assigned

        gk_slot = f"{side}_1"
        goalkeeper = self._select_goalkeeper_detection(side, detections)
        remaining_detections = detections
        if goalkeeper is not None:
            assigned[gk_slot] = goalkeeper
            remaining_detections = [det for det in detections if det["raw_id"] != goalkeeper["raw_id"]]

        field_template = self._aligned_side_template(side, remaining_detections)
        remaining_slots = [slot for slot in slot_names if slot not in assigned]
        if remaining_slots and remaining_detections:
            cost_matrix = np.zeros((len(remaining_detections), len(remaining_slots)), dtype=np.float32)
            for row_idx, detection in enumerate(remaining_detections):
                for col_idx, slot_name in enumerate(remaining_slots):
                    slot_idx = int(slot_name.split("_")[-1]) - 1
                    template_xy = field_template[slot_idx]
                    previous_xy = state["xy"].get(slot_name)
                    previous_raw_id = state["raw_id"].get(slot_name)
                    continuity_cost = 0.0
                    if previous_xy is not None:
                        continuity_cost = float(np.linalg.norm(np.asarray(detection["xy"]) - np.asarray(previous_xy)))
                    template_cost = float(np.linalg.norm(np.asarray(detection["xy"]) - template_xy))
                    raw_bonus = 0.0
                    if previous_raw_id is not None and str(previous_raw_id) == str(detection["raw_id"]):
                        raw_bonus = -4.0
                    weight_prev = 0.75 if previous_xy is not None else 0.0
                    weight_template = 1.0 - weight_prev if previous_xy is not None else 1.0
                    cost_matrix[row_idx, col_idx] = (weight_prev * continuity_cost) + (weight_template * template_cost) + raw_bonus

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
                assigned[remaining_slots[col_idx]] = remaining_detections[row_idx]
        return assigned

    def _aligned_side_template(self, side: str, detections: list[dict[str, Any]]) -> np.ndarray:
        pitch_template = self._team_template_for_side(side)
        if not detections:
            return pitch_template
        observed_arr = np.asarray([det["xy"] for det in detections], dtype=np.float32)
        template_center = np.median(pitch_template, axis=0)
        observed_center = np.median(observed_arr, axis=0)
        shift = observed_center - template_center
        aligned = pitch_template + shift
        aligned[:, 0] = np.clip(aligned[:, 0], 0.0, self.config.pitch_length_m)
        aligned[:, 1] = np.clip(aligned[:, 1], 0.0, self.config.pitch_width_m)
        return aligned

    def _team_template_for_side(self, side: str) -> np.ndarray:
        pitch_template = HOME_TEAM_TEMPLATE.copy()
        if side == "away":
            pitch_template[:, 0] = self.config.pitch_length_m - pitch_template[:, 0]
        return pitch_template

    def _select_goalkeeper_detection(
        self,
        side: str,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        goalkeeper_candidates = [
            detection
            for detection in detections
            if detection["class_name"] == "goalkeeper" or detection["role"] == "POR"
        ]
        candidates = goalkeeper_candidates if goalkeeper_candidates else detections
        reverse = side == "away"
        return sorted(candidates, key=lambda item: float(item["xy"][0]), reverse=reverse)[0] if candidates else None

    def _build_slot_track_arrays(
        self,
        tracks: Mapping[str, Any],
        frame_count: int,
        raw_to_slot: Mapping[str, str],
        ordered_slots: list[str],
        class_names: tuple[str, ...] = ("player", "goalkeeper"),
        max_speed_mps: float | None = None,
    ) -> dict[str, pd.DataFrame]:
        slot_frames = {
            slot_name: pd.DataFrame(index=np.arange(frame_count), columns=["x", "y"], dtype=np.float32)
            for slot_name in ordered_slots
        }
        for class_name in class_names:
            for frame_id, frame_map in enumerate(tracks.get(class_name, [])):
                if frame_id >= frame_count or not isinstance(frame_map, Mapping):
                    continue
                for raw_id, payload in frame_map.items():
                    slot_name = raw_to_slot.get(str(raw_id))
                    if slot_name is None:
                        continue
                    if not self._should_use_payload_for_slot_tracking(payload):
                        continue
                    field_position = self._safe_field_position(payload.get("field_position_m"))
                    if field_position is None:
                        continue
                    slot_frames[slot_name].loc[frame_id, ["x", "y"]] = field_position

        for slot_name, slot_df in slot_frames.items():
            slot_frames[slot_name] = self._stabilize_xy(
                self._interpolate_xy(slot_df),
                max_speed_mps=float(max_speed_mps) if max_speed_mps is not None else self.config.player_outlier_speed_mps,
            )
        return slot_frames

    def _interpolate_xy(self, slot_df: pd.DataFrame) -> pd.DataFrame:
        result = slot_df.copy()
        result[["x", "y"]] = result[["x", "y"]].interpolate(method="linear", limit_direction="both")
        result["x"] = result["x"].clip(0.0, self.config.pitch_length_m)
        result["y"] = result["y"].clip(0.0, self.config.pitch_width_m)
        return result

    def _stabilize_xy(self, slot_df: pd.DataFrame, max_speed_mps: float) -> pd.DataFrame:
        result = slot_df.copy()
        passes = max(1, int(self.config.smoothing_passes))
        for _ in range(passes):
            result = self._replace_isolated_motion_outliers(result, max_speed_mps=max_speed_mps)
            result = self._smooth_xy(result)
            result = self._limit_step_jitter(result, max_speed_mps=max_speed_mps)
        result["x"] = result["x"].clip(0.0, self.config.pitch_length_m)
        result["y"] = result["y"].clip(0.0, self.config.pitch_width_m)
        return result

    def _replace_isolated_motion_outliers(self, slot_df: pd.DataFrame, max_speed_mps: float) -> pd.DataFrame:
        result = slot_df.copy()
        dt = 1.0 / float(self.config.fps)
        max_step = float(max_speed_mps) * dt
        xy = result[["x", "y"]].to_numpy(dtype=np.float32, copy=True)
        if len(xy) < 3:
            return result
        for idx in range(1, len(xy) - 1):
            prev_xy = xy[idx - 1]
            curr_xy = xy[idx]
            next_xy = xy[idx + 1]
            if not (np.all(np.isfinite(prev_xy)) and np.all(np.isfinite(curr_xy)) and np.all(np.isfinite(next_xy))):
                continue
            prev_jump = float(np.linalg.norm(curr_xy - prev_xy))
            next_jump = float(np.linalg.norm(next_xy - curr_xy))
            bridge_jump = float(np.linalg.norm(next_xy - prev_xy))
            if prev_jump <= max_step or next_jump <= max_step:
                continue
            if bridge_jump > max_step:
                continue
            xy[idx] = (prev_xy + next_xy) * 0.5
        result.loc[:, ["x", "y"]] = xy
        return result

    def _smooth_xy(self, slot_df: pd.DataFrame) -> pd.DataFrame:
        result = slot_df.copy()
        window = max(1, int(self.config.smoothing_window))
        if window <= 1:
            return result
        for axis in ("x", "y"):
            series = result[axis]
            series = series.rolling(window=window, min_periods=1, center=bool(self.config.smoothing_center)).median()
            series = series.interpolate(method="linear", limit_direction="both").ffill().bfill()
            savgol_window = self._resolve_savgol_window(len(series))
            if savgol_window is not None:
                series_np = series.to_numpy(dtype=np.float32)
                if np.isfinite(series_np).all():
                    series = pd.Series(
                        savgol_filter(
                            series_np,
                            window_length=savgol_window,
                            polyorder=min(int(self.config.savgol_polyorder), savgol_window - 1),
                            mode="interp",
                        ),
                        index=series.index,
                        dtype=np.float32,
                    )
            series = series.ewm(alpha=0.35, adjust=False).mean()
            result[axis] = series
        return result

    def _resolve_savgol_window(self, series_len: int) -> int | None:
        if series_len < 3:
            return None
        window = max(3, int(self.config.savgol_window))
        if window % 2 == 0:
            window += 1
        if window > series_len:
            window = series_len if series_len % 2 == 1 else series_len - 1
        if window < 3:
            return None
        return window

    def _limit_step_jitter(self, slot_df: pd.DataFrame, max_speed_mps: float) -> pd.DataFrame:
        result = slot_df.copy()
        xy = result[["x", "y"]].to_numpy(dtype=np.float32, copy=True)
        if len(xy) < 2:
            return result
        dt = 1.0 / float(self.config.fps)
        max_step = float(max_speed_mps) * dt
        soft_step = max_step * 0.75
        for idx in range(1, len(xy)):
            prev_xy = xy[idx - 1]
            curr_xy = xy[idx]
            if not (np.all(np.isfinite(prev_xy)) and np.all(np.isfinite(curr_xy))):
                continue
            delta = curr_xy - prev_xy
            step = float(np.linalg.norm(delta))
            if step <= soft_step or step <= 1e-6:
                continue
            scale = soft_step / step
            xy[idx] = prev_xy + (delta * scale)
        result.loc[:, ["x", "y"]] = xy
        return result

    def _fill_team_templates(self, slot_tracks: dict[str, pd.DataFrame], side: str) -> None:
        pitch_template = self._team_template_for_side(side)
        side_slots = [f"{side}_{idx}" for idx in range(1, self.config.expected_players_per_team + 1)]
        for frame_id in range(len(next(iter(slot_tracks.values())))):
            observed_xy = []
            template_xy = []
            for idx, slot_name in enumerate(side_slots):
                row = slot_tracks[slot_name].iloc[frame_id]
                if np.isfinite(row["x"]) and np.isfinite(row["y"]):
                    observed_xy.append([row["x"], row["y"]])
                    template_xy.append(pitch_template[idx])
            shift = np.zeros(2, dtype=np.float32)
            if observed_xy:
                observed_arr = np.asarray(observed_xy, dtype=np.float32)
                template_arr = np.asarray(template_xy, dtype=np.float32)
                shift = np.mean(observed_arr - template_arr, axis=0)
            aligned_template = pitch_template + shift
            aligned_template[:, 0] = np.clip(aligned_template[:, 0], 0.0, self.config.pitch_length_m)
            aligned_template[:, 1] = np.clip(aligned_template[:, 1], 0.0, self.config.pitch_width_m)
            for idx, slot_name in enumerate(side_slots):
                row = slot_tracks[slot_name].iloc[frame_id]
                if np.isfinite(row["x"]) and np.isfinite(row["y"]):
                    continue
                slot_tracks[slot_name].loc[frame_id, ["x", "y"]] = aligned_template[idx]

    def _fill_referee_templates(self, referee_tracks: dict[str, pd.DataFrame]) -> None:
        if not referee_tracks:
            return
        ordered_slots = sorted(referee_tracks.keys())
        for frame_id in range(len(next(iter(referee_tracks.values())))):
            observed_xy = []
            template_xy = []
            for idx, slot_name in enumerate(ordered_slots):
                row = referee_tracks[slot_name].iloc[frame_id]
                if np.isfinite(row["x"]) and np.isfinite(row["y"]):
                    observed_xy.append([row["x"], row["y"]])
                    template_xy.append(REFEREE_TEMPLATE[idx])
            shift = np.zeros(2, dtype=np.float32)
            if observed_xy:
                observed_arr = np.asarray(observed_xy, dtype=np.float32)
                template_arr = np.asarray(template_xy, dtype=np.float32)
                shift = np.mean(observed_arr - template_arr, axis=0)
            aligned_template = REFEREE_TEMPLATE + shift
            aligned_template[:, 0] = np.clip(aligned_template[:, 0], 0.0, self.config.pitch_length_m)
            aligned_template[:, 1] = np.clip(aligned_template[:, 1], 0.0, self.config.pitch_width_m)
            for idx, slot_name in enumerate(ordered_slots):
                row = referee_tracks[slot_name].iloc[frame_id]
                if np.isfinite(row["x"]) and np.isfinite(row["y"]):
                    continue
                referee_tracks[slot_name].loc[frame_id, ["x", "y"]] = aligned_template[idx]

    def _build_ball_and_carrier_series(
        self,
        tracks: Mapping[str, Any],
        frame_count: int,
        raw_to_slot: Mapping[str, str],
        slot_tracks: Mapping[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        del tracks, raw_to_slot, slot_tracks
        ball_df = pd.DataFrame(index=np.arange(frame_count), columns=["x", "y"], dtype=np.float32)
        carriers = pd.Series([None] * frame_count, dtype=object)
        owning_teams = pd.Series([None] * frame_count, dtype=object)
        return ball_df, carriers, owning_teams

    @staticmethod
    def _first_payload(class_frames: Any, frame_id: int) -> Mapping[str, Any] | None:
        if not isinstance(class_frames, list) or frame_id >= len(class_frames):
            return None
        frame_map = class_frames[frame_id]
        if not isinstance(frame_map, Mapping) or not frame_map:
            return None
        return next(iter(frame_map.values()))

    @staticmethod
    def _bbox_center(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _bbox_footpoint(bbox: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
        if bbox is None:
            return None
        x1, _, x2, y2 = bbox
        return (x1 + x2) / 2.0, y2

    def _nearest_visible_slot(
        self,
        ball_center: tuple[float, float] | None,
        tracks: Mapping[str, Any],
        frame_id: int,
        raw_to_slot: Mapping[str, str],
    ) -> str | None:
        if ball_center is None:
            return None
        best_slot = None
        best_distance = None
        for class_name in ("player", "goalkeeper"):
            frames = tracks.get(class_name, [])
            if frame_id >= len(frames):
                continue
            frame_map = frames[frame_id]
            if not isinstance(frame_map, Mapping):
                continue
            for raw_id, payload in frame_map.items():
                slot_name = raw_to_slot.get(str(raw_id))
                if slot_name is None:
                    continue
                bbox = self._safe_bbox(payload.get("bbox"))
                footpoint = self._bbox_footpoint(bbox)
                if footpoint is None:
                    continue
                distance = math.dist(ball_center, footpoint)
                inside_bbox = self._point_inside_bbox(ball_center, bbox, self.config.bbox_inside_padding_px)
                if not inside_bbox and distance > self.config.carrier_max_distance_px:
                    continue
                if best_distance is None or inside_bbox or distance < best_distance:
                    best_slot = slot_name
                    best_distance = distance
                    if inside_bbox:
                        return best_slot
        return best_slot

    @staticmethod
    def _point_inside_bbox(
        point: tuple[float, float],
        bbox: tuple[float, float, float, float] | None,
        padding: float,
    ) -> bool:
        if bbox is None:
            return False
        x, y = point
        x1, y1, x2, y2 = bbox
        return (x1 - padding) <= x <= (x2 + padding) and (y1 - padding) <= y <= (y2 + padding)

    def _compute_motion_features(self, xy_df: pd.DataFrame) -> pd.DataFrame:
        result = xy_df.copy()
        dt = 1.0 / float(self.config.fps)
        result["vx"] = result["x"].diff().fillna(0.0) / dt
        result["vy"] = result["y"].diff().fillna(0.0) / dt
        result["speed"] = np.sqrt((result["vx"] ** 2) + (result["vy"] ** 2))
        result["accel"] = result["speed"].diff().fillna(0.0) / dt
        return result

    @staticmethod
    def _team_from_slot(slot_name: Any) -> str | None:
        if slot_name is None or (isinstance(slot_name, float) and np.isnan(slot_name)):
            return None
        slot_name = str(slot_name)
        if slot_name.startswith("home_"):
            return "home"
        if slot_name.startswith("away_"):
            return "away"
        return None
