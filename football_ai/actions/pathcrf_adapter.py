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

        team_records: dict[str, list[TrackRecord]] = {}
        for record in person_records:
            team_name = record.stable_team_name()
            if team_name:
                team_records.setdefault(team_name, []).append(record)

        ordered_teams = self._order_teams_by_side(team_records)
        side_by_team = {}
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
        remaining.sort(
            key=lambda item: (
                item.first_frame,
                item.median_x(self.config.pitch_length_m / 2.0),
                int(item.raw_track_id),
            )
        )
        for idx, record in enumerate(remaining[: self.config.expected_players_per_team - 1], start=2):
            assignments[f"{side}_{idx}"] = record.raw_track_id
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
        person_assignments: Mapping[str, str],
        referee_assignments: Mapping[str, str],
    ) -> pd.DataFrame:
        raw_to_slot = {raw_id: slot for slot, raw_id in person_assignments.items()}
        raw_ref_to_slot = {raw_id: slot for slot, raw_id in referee_assignments.items()}
        all_person_slots = [f"home_{idx}" for idx in range(1, 12)] + [f"away_{idx}" for idx in range(1, 12)]
        all_ref_slots = [f"referee_{idx}" for idx in range(1, self.config.expected_referees + 1)]

        slot_tracks = self._build_slot_track_arrays(tracks, frame_count, raw_to_slot, all_person_slots)
        referee_tracks = self._build_slot_track_arrays(tracks, frame_count, raw_ref_to_slot, all_ref_slots, class_names=("referee",))
        self._fill_team_templates(slot_tracks, "home")
        self._fill_team_templates(slot_tracks, "away")
        self._fill_referee_templates(referee_tracks)

        ball_df, carrier_series = self._build_ball_and_carrier_series(
            tracks=tracks,
            frame_count=frame_count,
            raw_to_slot=raw_to_slot,
            slot_tracks=slot_tracks,
        )

        state_df = pd.DataFrame(
            {
                "frame_id": np.arange(frame_count, dtype=np.int32),
                "period_id": np.ones(frame_count, dtype=np.int16),
                "timestamp": np.arange(frame_count, dtype=np.float32) / float(self.config.fps),
                "phase_id": np.ones(frame_count, dtype=np.int16),
                "episode_id": np.ones(frame_count, dtype=np.int16),
                "ball_state": np.full(frame_count, "alive", dtype=object),
                "ball_owning_team_id": carrier_series.apply(self._team_from_slot),
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

    def _build_slot_track_arrays(
        self,
        tracks: Mapping[str, Any],
        frame_count: int,
        raw_to_slot: Mapping[str, str],
        ordered_slots: list[str],
        class_names: tuple[str, ...] = ("player", "goalkeeper"),
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
                    field_position = self._safe_field_position(payload.get("field_position_m"))
                    if field_position is None:
                        continue
                    slot_frames[slot_name].loc[frame_id, ["x", "y"]] = field_position

        for slot_name, slot_df in slot_frames.items():
            slot_frames[slot_name] = self._interpolate_xy(slot_df)
        return slot_frames

    def _interpolate_xy(self, slot_df: pd.DataFrame) -> pd.DataFrame:
        result = slot_df.copy()
        result[["x", "y"]] = result[["x", "y"]].interpolate(
            method="linear",
            limit_direction="both",
        )
        return result

    def _fill_team_templates(self, slot_tracks: dict[str, pd.DataFrame], side: str) -> None:
        pitch_template = HOME_TEAM_TEMPLATE.copy()
        if side == "away":
            pitch_template[:, 0] = self.config.pitch_length_m - pitch_template[:, 0]

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
    ) -> tuple[pd.DataFrame, pd.Series]:
        ball_df = pd.DataFrame(index=np.arange(frame_count), columns=["x", "y"], dtype=np.float32)
        carriers: list[str | None] = []
        previous_carrier: str | None = None
        previous_ball_xy: tuple[float, float] | None = None

        for frame_id in range(frame_count):
            ball_payload = self._first_payload(tracks.get("ball", []), frame_id)
            ball_center = self._bbox_center(self._safe_bbox(ball_payload.get("bbox") if ball_payload else None))
            carrier_slot = self._nearest_visible_slot(ball_center, tracks, frame_id, raw_to_slot)

            if carrier_slot is None:
                carrier_slot = previous_carrier

            if carrier_slot is not None:
                carrier_xy = slot_tracks[carrier_slot].iloc[frame_id]
                ball_xy = (float(carrier_xy["x"]), float(carrier_xy["y"]))
            elif previous_ball_xy is not None:
                ball_xy = previous_ball_xy
            else:
                ball_xy = (self.config.pitch_length_m / 2.0, self.config.pitch_width_m / 2.0)

            carriers.append(carrier_slot)
            ball_df.loc[frame_id, ["x", "y"]] = ball_xy
            previous_carrier = carrier_slot
            previous_ball_xy = ball_xy

        return ball_df, pd.Series(carriers, dtype=object)

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
