from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PossessionConfig:
    enabled: bool = True
    strict_control_distance_px: float = 60.0
    strict_control_distance_ratio_to_height: float = 0.65
    loose_control_distance_px: float = 95.0
    loose_control_distance_ratio_to_height: float = 1.0
    continuation_distance_px: float = 35.0
    immediate_opponent_switch_distance_px: float = 24.0
    opponent_switch_confirmation_frames: int = 2
    opponent_switch_window_frames: int = 4
    slow_ball_speed_px: float = 14.0
    same_player_control_speed_px: float = 18.0
    speed_drop_threshold_px: float = 8.0
    direction_change_threshold_deg: float = 35.0
    opponent_takeover_margin_px: float = 12.0
    ball_missing_release_frames: int = 10
    touch_timeout_frames: int = 90

    @classmethod
    def from_mapping(cls, raw_config: Mapping[str, Any] | None) -> "PossessionConfig":
        if not isinstance(raw_config, Mapping):
            return cls()
        payload = {}
        for field_name in cls.__dataclass_fields__.keys():
            if field_name in raw_config:
                payload[field_name] = raw_config[field_name]
        return cls(**payload)


@dataclass(frozen=True)
class Candidate:
    track_id: Any
    class_name: str
    team_id: str | None
    distance_px: float
    player_height_px: float
    inside_bbox: bool


class TeamPossessionEstimator:
    def __init__(self, config: PossessionConfig | None = None):
        self.config = config or PossessionConfig()
        self.current_team: str | None = None
        self.current_player: Any | None = None
        self.last_touch_frame = -10_000
        self.last_ball_frame = -10_000
        self.pending_switch_team: str | None = None
        self.pending_switch_player: Any | None = None
        self.pending_switch_count = 0
        self.pending_switch_last_frame = -10_000
        self.previous_ball_center: tuple[float, float] | None = None
        self.previous_ball_speed: float | None = None
        self.previous_ball_vector: tuple[float, float] | None = None

    @staticmethod
    def _ball_center(frame_ball_tracks: Mapping[str, Any]) -> tuple[float, float] | None:
        if not isinstance(frame_ball_tracks, Mapping) or not frame_ball_tracks:
            return None
        _, data = next(iter(frame_ball_tracks.items()))
        bbox = data.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _player_footpoint(track_data: Mapping[str, Any]) -> tuple[float, float] | None:
        bbox = track_data.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        x1, _, x2, y2 = [float(v) for v in bbox[:4]]
        return (x1 + x2) / 2.0, y2

    @staticmethod
    def _player_height_px(track_data: Mapping[str, Any]) -> float | None:
        bbox = track_data.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        y1, y2 = float(bbox[1]), float(bbox[3])
        return max(1.0, y2 - y1)

    @staticmethod
    def _ball_inside_expanded_bbox(
        ball_center: tuple[float, float],
        track_data: Mapping[str, Any],
    ) -> bool:
        bbox = track_data.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bx, by = ball_center
        return (x1 - 8.0) <= bx <= (x2 + 8.0) and (y1 - 8.0) <= by <= (y2 + 16.0)

    def _frame_candidates(
        self,
        frame_tracks: Mapping[str, Mapping[Any, Any]],
        ball_center: tuple[float, float] | None,
    ) -> list[Candidate]:
        if ball_center is None:
            return []
        candidates: list[Candidate] = []
        for class_name in ("player", "goalkeeper"):
            class_map = frame_tracks.get(class_name, {})
            if not isinstance(class_map, Mapping):
                continue
            for track_id, track_data in class_map.items():
                if not isinstance(track_data, Mapping):
                    continue
                footpoint = self._player_footpoint(track_data)
                player_height = self._player_height_px(track_data)
                if footpoint is None or player_height is None:
                    continue
                distance = math.dist(ball_center, footpoint)
                candidates.append(
                    Candidate(
                        track_id=track_id,
                        class_name=class_name,
                        team_id=track_data.get("team"),
                        distance_px=float(distance),
                        player_height_px=float(player_height),
                        inside_bbox=self._ball_inside_expanded_bbox(ball_center, track_data),
                    )
                )
        candidates.sort(key=lambda candidate: candidate.distance_px)
        return candidates

    @staticmethod
    def _direction_change_deg(
        previous_vector: tuple[float, float] | None,
        current_vector: tuple[float, float] | None,
    ) -> float | None:
        if previous_vector is None or current_vector is None:
            return None
        n1 = math.hypot(*previous_vector)
        n2 = math.hypot(*current_vector)
        if n1 <= 1e-6 or n2 <= 1e-6:
            return None
        cosine = max(
            -1.0,
            min(
                1.0,
                (
                    previous_vector[0] * current_vector[0]
                    + previous_vector[1] * current_vector[1]
                )
                / (n1 * n2),
            ),
        )
        return math.degrees(math.acos(cosine))

    def update_frame(
        self,
        frame_id: int,
        frame_tracks: Mapping[str, Mapping[Any, Any]],
    ) -> dict[str, Any]:
        if not self.config.enabled:
            return {
                "team_id": None,
                "player_id": None,
                "reason": "disabled",
                "ball_detected": False,
            }

        ball_center = self._ball_center(frame_tracks.get("ball", {}))
        if ball_center is not None:
            self.last_ball_frame = frame_id

        ball_speed = None
        current_ball_vector = None
        if ball_center is not None and self.previous_ball_center is not None:
            current_ball_vector = (
                ball_center[0] - self.previous_ball_center[0],
                ball_center[1] - self.previous_ball_center[1],
            )
            ball_speed = math.hypot(*current_ball_vector)
        direction_change = self._direction_change_deg(
            self.previous_ball_vector,
            current_ball_vector,
        )
        speed_drop = (
            self.previous_ball_speed - ball_speed
            if self.previous_ball_speed is not None and ball_speed is not None
            else None
        )

        if (
            self.pending_switch_team is not None
            and (frame_id - self.pending_switch_last_frame) > self.config.opponent_switch_window_frames
        ):
            self.pending_switch_team = None
            self.pending_switch_player = None
            self.pending_switch_count = 0

        candidates = self._frame_candidates(frame_tracks, ball_center)
        team_candidates = [candidate for candidate in candidates if candidate.team_id]
        best = team_candidates[0] if team_candidates else None
        second = team_candidates[1] if len(team_candidates) > 1 else None

        touch_candidate: Candidate | None = None
        touch_reason: str | None = None
        possession_reason = "last_touch_hold" if self.current_team is not None else "unknown"

        if best is not None:
            strict_threshold = min(
                self.config.strict_control_distance_px,
                best.player_height_px * self.config.strict_control_distance_ratio_to_height,
            )
            loose_threshold = min(
                self.config.loose_control_distance_px,
                best.player_height_px * self.config.loose_control_distance_ratio_to_height,
            )
            strict_contact = best.distance_px <= strict_threshold
            loose_contact = best.distance_px <= loose_threshold
            separation = (
                second.distance_px - best.distance_px if second is not None else float("inf")
            )
            motion_touch = (
                (ball_speed is not None and ball_speed <= self.config.slow_ball_speed_px)
                or (speed_drop is not None and speed_drop >= self.config.speed_drop_threshold_px)
                or (
                    direction_change is not None
                    and direction_change >= self.config.direction_change_threshold_deg
                )
            )
            start_touch = self.current_team is None and strict_contact and (
                motion_touch or best.inside_bbox
            )
            same_player_control = (
                self.current_player == best.track_id
                and loose_contact
                and (
                    (ball_speed is not None and ball_speed <= self.config.same_player_control_speed_px)
                    or best.distance_px <= self.config.continuation_distance_px
                )
            )
            opponent_takeover_candidate = (
                self.current_team is not None
                and best.team_id != self.current_team
                and strict_contact
                and separation >= self.config.opponent_takeover_margin_px
                and motion_touch
            )
            immediate_opponent_switch = opponent_takeover_candidate and (
                best.inside_bbox
                or best.distance_px <= self.config.immediate_opponent_switch_distance_px
            )
            confirmed_pending_switch = False

            if opponent_takeover_candidate and not immediate_opponent_switch:
                same_pending_team = (
                    self.pending_switch_team == best.team_id
                    and (frame_id - self.pending_switch_last_frame)
                    <= self.config.opponent_switch_window_frames
                )
                if same_pending_team:
                    self.pending_switch_count += 1
                else:
                    self.pending_switch_team = best.team_id
                    self.pending_switch_player = best.track_id
                    self.pending_switch_count = 1
                self.pending_switch_last_frame = frame_id
                confirmed_pending_switch = (
                    self.pending_switch_count >= self.config.opponent_switch_confirmation_frames
                )
            elif best.team_id == self.current_team or self.current_team is None:
                self.pending_switch_team = None
                self.pending_switch_player = None
                self.pending_switch_count = 0

            if immediate_opponent_switch:
                touch_candidate = best
                touch_reason = "opponent_touch"
            elif confirmed_pending_switch:
                touch_candidate = best
                touch_reason = "opponent_touch_confirmed"
            elif same_player_control:
                touch_candidate = best
                touch_reason = "same_player_control"
            elif start_touch:
                touch_candidate = best
                touch_reason = "start_touch"
            elif best.inside_bbox and strict_contact:
                touch_candidate = best
                touch_reason = "inside_bbox_touch"
            elif strict_contact and motion_touch:
                touch_candidate = best
                touch_reason = "motion_touch"

        if touch_candidate is not None:
            self.current_team = touch_candidate.team_id
            self.current_player = touch_candidate.track_id
            self.last_touch_frame = frame_id
            possession_reason = touch_reason or "touch"
            self.pending_switch_team = None
            self.pending_switch_player = None
            self.pending_switch_count = 0
        else:
            if self.current_team is not None:
                if (
                    frame_id - self.last_touch_frame > self.config.touch_timeout_frames
                    or frame_id - self.last_ball_frame > self.config.ball_missing_release_frames
                ):
                    self.current_team = None
                    self.current_player = None
                    possession_reason = "timeout_release"
                else:
                    possession_reason = "last_touch_hold"
            else:
                possession_reason = "unknown"

        self.previous_ball_vector = current_ball_vector
        self.previous_ball_speed = ball_speed
        if ball_center is not None:
            self.previous_ball_center = ball_center

        return {
            "team_id": self.current_team,
            "player_id": self.current_player,
            "reason": possession_reason,
            "ball_detected": ball_center is not None,
            "nearest_track_id": best.track_id if best is not None else None,
            "nearest_team_id": best.team_id if best is not None else None,
        }
