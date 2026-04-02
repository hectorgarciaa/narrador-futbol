from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)


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
class PathCRFAdapterConfig:
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
    output_dir: Path = Path("football_ai/actions/pathcrf/data/narrador/tracking_processed")


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
class ConversionSummary:
    track_count_by_class: dict[str, int]
    person_slot_assignments: dict[str, str]
    referee_slot_assignments: dict[str, str]
    synthetic_home_slots: list[str]
    synthetic_away_slots: list[str]
    synthetic_referee_slots: list[str]
    frames: int
    output_parquet: str


class PathCRFTracksAdapter:
    def __init__(self, config: PathCRFAdapterConfig | None = None):
        self.config = config or PathCRFAdapterConfig()

    def convert_file(self, tracks_path: str | Path, output_path: str | Path | None = None) -> tuple[Path, ConversionSummary]:
        tracks_path = Path(tracks_path)
        with tracks_path.open("r", encoding="utf-8") as f:
            tracks = json.load(f)

        frame_count = self._infer_frame_count(tracks)
        if frame_count <= 0:
            raise ValueError(f"No se encontraron frames en {tracks_path}")

        records = self._build_track_records(tracks, frame_count)
        person_assignments, referee_assignments, synthetic_slots = self._assign_slots(records)
        frame_df = self._build_tracking_dataframe(
            tracks=tracks,
            frame_count=frame_count,
            records=records,
            person_assignments=person_assignments,
            referee_assignments=referee_assignments,
        )

        target_path = Path(output_path) if output_path is not None else self._default_output_path(tracks_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        frame_df.to_parquet(target_path, index=False)

        summary = ConversionSummary(
            track_count_by_class={key: len(value) for key, value in records.items()},
            person_slot_assignments={raw_id: slot for slot, raw_id in person_assignments.items()},
            referee_slot_assignments={raw_id: slot for slot, raw_id in referee_assignments.items()},
            synthetic_home_slots=synthetic_slots["home"],
            synthetic_away_slots=synthetic_slots["away"],
            synthetic_referee_slots=synthetic_slots["referee"],
            frames=frame_count,
            output_parquet=str(target_path),
        )

        self._write_summary(target_path, summary)
        logger.info("PathCRF parquet generated: %s", target_path)
        return target_path, summary

    def _default_output_path(self, tracks_path: Path) -> Path:
        stem = self._sanitize_stem(tracks_path.stem.replace("_tracks", ""))
        return self.config.output_dir / f"{stem}.parquet"

    @staticmethod
    def _sanitize_stem(raw_stem: str) -> str:
        stem = re.sub(r"\s+", "_", str(raw_stem).strip())
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("._-")
        return stem or "tracks"

    def _write_summary(self, target_path: Path, summary: ConversionSummary) -> None:
        summary_path = target_path.with_suffix(".summary.json")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _infer_frame_count(tracks: Mapping[str, Any]) -> int:
        classes = ("player", "goalkeeper", "referee", "ball")
        return max((len(tracks.get(class_name, [])) for class_name in classes), default=0)

    def _build_track_records(self, tracks: Mapping[str, Any], frame_count: int) -> dict[str, dict[str, TrackRecord]]:
        records: dict[str, dict[str, TrackRecord]] = {
            "player": {},
            "goalkeeper": {},
            "referee": {},
            "ball": {},
        }
        for class_name in records.keys():
            frames = tracks.get(class_name, [])
            for frame_id in range(min(frame_count, len(frames))):
                frame_map = frames[frame_id]
                if not isinstance(frame_map, Mapping):
                    continue
                for raw_id, payload in frame_map.items():
                    if class_name in {"player", "goalkeeper", "referee"} and not self._should_use_payload_for_slot_tracking(payload):
                        continue
                    record = records[class_name].setdefault(
                        str(raw_id),
                        TrackRecord(raw_track_id=str(raw_id), class_name=class_name),
                    )
                    record.observations.append(
                        TrackObservation(
                            frame_id=frame_id,
                            class_name=class_name,
                            team_name=self._normalize_team_name(payload.get("team")),
                            predicted_role=self._normalize_role(payload.get("predicted_role") or payload.get("predicted_role_frame")),
                            bbox=self._safe_bbox(payload.get("bbox")),
                            field_position_m=self._safe_field_position(payload.get("field_position_m")),
                        )
                    )
        return records

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

        person_assignments = {}
        person_assignments.update(self._assign_team_slots("home", home_records))
        person_assignments.update(self._assign_team_slots("away", away_records))

        referee_assignments = {}
        for idx, record in enumerate(sorted(referee_records, key=lambda item: (item.first_frame, int(item.raw_track_id))), start=1):
            if idx > self.config.expected_referees:
                break
            referee_assignments[f"referee_{idx}"] = record.raw_track_id

        synthetic_slots = {
            "home": [f"home_{idx}" for idx in range(1, self.config.expected_players_per_team + 1) if f"home_{idx}" not in person_assignments],
            "away": [f"away_{idx}" for idx in range(1, self.config.expected_players_per_team + 1) if f"away_{idx}" not in person_assignments],
            "referee": [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1) if f"referee_{idx}" not in referee_assignments],
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

    def _build_tracking_dataframe(
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
        _, side_by_team = self._group_person_records_by_side(
            [*records["player"].values(), *records["goalkeeper"].values()]
        )

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
            slot_features = self._compute_motion_features(slot_df)
            output_df = pd.concat([output_df, slot_features.add_prefix(f"{slot_name}_")], axis=1)

        for slot_name, slot_df in referee_tracks.items():
            slot_features = self._compute_motion_features(slot_df)
            output_df = pd.concat([output_df, slot_features.add_prefix(f"{slot_name}_")], axis=1)

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
                            "role": self._normalize_role(
                                payload.get("predicted_role") or payload.get("predicted_role_frame")
                            ),
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
                    cost_matrix[row_idx, col_idx] = (
                        (weight_prev * continuity_cost)
                        + (weight_template * template_cost)
                        + raw_bonus
                    )

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
        return sorted(
            candidates,
            key=lambda item: float(item["xy"][0]),
            reverse=reverse,
        )[0] if candidates else None

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
                max_speed_mps=(
                    float(max_speed_mps)
                    if max_speed_mps is not None
                    else self.config.player_outlier_speed_mps
                ),
            )
        return slot_frames

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

    def _interpolate_xy(self, slot_df: pd.DataFrame) -> pd.DataFrame:
        result = slot_df.copy()
        result[["x", "y"]] = result[["x", "y"]].interpolate(
            method="linear",
            limit_direction="both",
        )
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

    def _replace_isolated_motion_outliers(
        self,
        slot_df: pd.DataFrame,
        max_speed_mps: float,
    ) -> pd.DataFrame:
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
            if not (
                np.all(np.isfinite(prev_xy))
                and np.all(np.isfinite(curr_xy))
                and np.all(np.isfinite(next_xy))
            ):
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
        ball_df = pd.DataFrame(index=np.arange(frame_count), columns=["x", "y"], dtype=np.float32)
        carriers: list[str | None] = []
        owning_teams: list[str | None] = []
        previous_carrier: str | None = None
        previous_team: str | None = None
        previous_ball_xy: tuple[float, float] | None = None

        for frame_id in range(frame_count):
            ball_payload = self._first_payload(tracks.get("ball", []), frame_id)
            ball_center = self._bbox_center(self._safe_bbox(ball_payload.get("bbox") if ball_payload else None))
            ball_field_position = self._safe_field_position(
                ball_payload.get("field_position_m") if isinstance(ball_payload, Mapping) else None
            )
            carrier_slot = None
            if isinstance(ball_payload, Mapping):
                owning_player_id = ball_payload.get("player_id")
                if owning_player_id is None:
                    owning_player_id = ball_payload.get("ball_owning_player_id")
                if owning_player_id is not None:
                    carrier_slot = raw_to_slot.get(str(owning_player_id))
            if carrier_slot is None:
                carrier_slot = self._nearest_visible_slot(ball_center, tracks, frame_id, raw_to_slot)

            if carrier_slot is None:
                carrier_slot = previous_carrier

            carrier_xy = None
            if carrier_slot is not None:
                carrier_xy = slot_tracks[carrier_slot].iloc[frame_id]
                ball_xy = (float(carrier_xy["x"]), float(carrier_xy["y"]))
                if not np.all(np.isfinite(ball_xy)):
                    ball_xy = None
                else:
                    carrier_xy = ball_xy
            if ball_field_position is not None:
                ball_xy = ball_field_position
            elif carrier_xy is not None:
                ball_xy = carrier_xy
            elif previous_ball_xy is not None:
                ball_xy = previous_ball_xy
            else:
                ball_xy = (self.config.pitch_length_m / 2.0, self.config.pitch_width_m / 2.0)

            owning_team = self._team_from_slot(carrier_slot)
            if owning_team is None:
                owning_team = previous_team

            carriers.append(carrier_slot)
            owning_teams.append(owning_team)
            ball_df.loc[frame_id, ["x", "y"]] = ball_xy
            previous_carrier = carrier_slot
            previous_team = owning_team
            previous_ball_xy = ball_xy

        ball_df = self._stabilize_xy(ball_df, max_speed_mps=self.config.ball_outlier_speed_mps)
        return ball_df, pd.Series(carriers, dtype=object), pd.Series(owning_teams, dtype=object)

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


def convert_tracks_json_to_pathcrf(
    tracks_path: str | Path,
    output_path: str | Path | None = None,
    config: PathCRFAdapterConfig | None = None,
) -> tuple[Path, ConversionSummary]:
    adapter = PathCRFTracksAdapter(config=config)
    return adapter.convert_file(tracks_path=tracks_path, output_path=output_path)
