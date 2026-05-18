from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Any, Callable

from .pathcrf_semantics import PITCH_CENTER_Y, PITCH_LENGTH_M, safe_float, team_id_from_player_id

TRACK_LOOKUP = Callable[[int | None, str | None], dict[str, Any] | None]

ATTACKING_ZONE_THRESHOLD_M = 70.0
MAX_SHOT_DISTANCE_M = 42.0
SHOT_MIN_DIRECTION_SIMILARITY = 0.9
SHOT_MIN_GOAL_PROGRESS_M = 3.0
SHOT_CENTRAL_LANE_DISTANCE_M = 18.0
MIN_DIRECTION_VOTE_DX_M = 1.0
ATTACK_DIRECTION_MEMORY = 12


@dataclass
class PossessionState:
    player_id: str | None = None
    track_id: str | None = None
    team_id: str | None = None


class OnlineSemanticPostprocessor:
    def __init__(self, *, fps: float = 25.0) -> None:
        self.fps = max(float(fps), 1e-6)
        self.reset()

    def reset(self) -> None:
        self._attack_direction_votes: dict[str, deque[float]] = {
            "home": deque(maxlen=ATTACK_DIRECTION_MEMORY),
            "away": deque(maxlen=ATTACK_DIRECTION_MEMORY),
        }
        self._possession = PossessionState()

    def process_events(
        self,
        events: list[dict[str, Any]],
        *,
        track_lookup: TRACK_LOOKUP,
    ) -> list[dict[str, Any]]:
        processed: list[dict[str, Any]] = []
        for event in sorted(events, key=self._sort_key):
            prepared = self._prepare_event(event, track_lookup=track_lookup)
            self._observe_attack_direction(prepared)
            semantic_event = self._apply_semantics(prepared)
            self._update_possession(semantic_event)
            processed.append(semantic_event)
        return processed

    @staticmethod
    def _sort_key(event: dict[str, Any]) -> tuple[int, int]:
        start_frame = _safe_int(event.get("start_frame"))
        frame_id = _safe_int(event.get("frame_id"))
        return (start_frame if start_frame is not None else frame_id if frame_id is not None else -1, frame_id or -1)

    def _prepare_event(
        self,
        event: dict[str, Any],
        *,
        track_lookup: TRACK_LOOKUP,
    ) -> dict[str, Any]:
        result = dict(event)
        frame_id = _safe_int(result.get("frame_id"))
        start_frame = _safe_int(result.get("start_frame"))
        end_frame = _safe_int(result.get("end_frame"))
        if frame_id is None:
            frame_id = start_frame
        if start_frame is None:
            start_frame = frame_id
        if end_frame is None:
            end_frame = frame_id

        raw_event_type = str(result.get("event_type_raw") or result.get("event_type") or "").strip() or "unknown"
        result["frame_id"] = frame_id
        result["start_frame"] = start_frame
        result["end_frame"] = end_frame
        result["event_type_raw"] = raw_event_type
        result["event_type_semantic"] = str(result.get("event_type_semantic") or raw_event_type)
        result["event_type"] = result["event_type_semantic"]
        result["semantic_source"] = str(result.get("semantic_source") or "rolling_postprocess")
        semantic_confidence = _safe_float(result.get("semantic_confidence"))
        result["semantic_confidence"] = 1.0 if semantic_confidence is None else semantic_confidence
        result["timestamp"] = f"{float(start_frame or 0) / self.fps:.3f}"

        player_id = _clean_id(result.get("player_id") or result.get("canonical_src"))
        receiver_id = _clean_id(result.get("receiver_id") or result.get("canonical_dst"))
        player_track_id = _clean_id(result.get("player_track_id"))
        receiver_track_id = _clean_id(result.get("receiver_track_id"))
        result["player_id"] = player_id
        result["receiver_id"] = receiver_id
        result["canonical_src"] = player_id
        result["canonical_dst"] = receiver_id
        result["player_track_id"] = player_track_id
        result["receiver_track_id"] = receiver_track_id

        player_payload = track_lookup(start_frame, player_track_id)
        receiver_payload = track_lookup(end_frame, receiver_track_id)
        result["_player_payload"] = player_payload
        result["_receiver_payload"] = receiver_payload
        result["_player_team_id"] = team_id_from_player_id(player_id)
        result["_receiver_team_id"] = team_id_from_player_id(receiver_id)
        result["_player_team_name"] = _clean_text((player_payload or {}).get("team"))
        result["_receiver_team_name"] = _clean_text((receiver_payload or {}).get("team"))
        result["_receiver_class_name"] = _clean_text((receiver_payload or {}).get("_class_name"))

        start_pos = _field_position(player_payload) or _event_position(result, "start")
        end_pos = _field_position(receiver_payload) or _event_position(result, "end")
        if end_pos is None and player_track_id:
            end_pos = _field_position(track_lookup(end_frame, player_track_id))
        result["start_x"] = start_pos[0] if start_pos is not None else _safe_float(result.get("start_x"))
        result["start_y"] = start_pos[1] if start_pos is not None else _safe_float(result.get("start_y"))
        result["end_x"] = end_pos[0] if end_pos is not None else _safe_float(result.get("end_x"))
        result["end_y"] = end_pos[1] if end_pos is not None else _safe_float(result.get("end_y"))
        return result

    def _observe_attack_direction(self, event: dict[str, Any]) -> None:
        raw_event_type = str(event.get("event_type_raw") or "").strip()
        team_id = event.get("_player_team_id")
        if raw_event_type != "kick" or team_id not in {"home", "away"}:
            return
        start_x = _safe_float(event.get("start_x"))
        end_x = _safe_float(event.get("end_x"))
        if start_x is None or end_x is None:
            return
        dx = end_x - start_x
        if abs(dx) < MIN_DIRECTION_VOTE_DX_M:
            return
        self._attack_direction_votes.setdefault(str(team_id), deque(maxlen=ATTACK_DIRECTION_MEMORY)).append(dx)

    def _team_attacks_right(self, team_id: str | None) -> bool:
        if team_id not in {"home", "away"}:
            return False
        votes = self._attack_direction_votes.get(team_id)
        if votes:
            return (sum(votes) / float(len(votes))) >= 0.0
        return team_id == "home"

    def _apply_semantics(self, event: dict[str, Any]) -> dict[str, Any]:
        raw_event_type = str(event.get("event_type_raw") or "").strip()
        team_id = event.get("_player_team_id")
        attack_direction_right = self._team_attacks_right(team_id)
        event["attack_direction_right"] = attack_direction_right

        shot_score = self._shot_score(event, attack_direction_right=attack_direction_right)
        event["shot_score_total"] = float(shot_score)
        event["is_shot_candidate"] = raw_event_type == "kick" and shot_score > 0.0
        event["is_pred_shot"] = False

        if raw_event_type == "kick" and shot_score >= 6.0:
            return self._apply_shot(event, shot_score=shot_score)

        if raw_event_type == "kick" and self._is_cross_team_recovery(event):
            return self._apply_robbery_from_pass(event)

        if raw_event_type == "control" and self._is_control_recovery(event):
            return self._apply_robbery_from_control(event)

        return self._finalize_event(event, semantic_type=raw_event_type, semantic_source="rolling_postprocess", confidence=1.0)

    def _shot_score(self, event: dict[str, Any], *, attack_direction_right: bool) -> float:
        if str(event.get("event_type_raw") or "").strip() != "kick":
            return 0.0
        start_x = _safe_float(event.get("start_x"))
        start_y = _safe_float(event.get("start_y"))
        end_x = _safe_float(event.get("end_x"))
        end_y = _safe_float(event.get("end_y"))
        team_id = event.get("_player_team_id")
        receiver_team_id = event.get("_receiver_team_id")
        if start_x is None or start_y is None or end_x is None or end_y is None or team_id not in {"home", "away"}:
            return 0.0

        goal_x = PITCH_LENGTH_M if attack_direction_right else 0.0
        goal_y = PITCH_CENTER_Y
        dist_start = hypot(goal_x - start_x, goal_y - start_y)
        dist_end = hypot(goal_x - end_x, goal_y - end_y)
        if dist_start > MAX_SHOT_DISTANCE_M:
            return 0.0

        in_attacking_zone = start_x >= ATTACKING_ZONE_THRESHOLD_M if attack_direction_right else start_x <= (PITCH_LENGTH_M - ATTACKING_ZONE_THRESHOLD_M)
        if not in_attacking_zone:
            return 0.0

        move_dx = end_x - start_x
        move_dy = end_y - start_y
        move_norm = hypot(move_dx, move_dy)
        goal_dx = goal_x - start_x
        goal_dy = goal_y - start_y
        goal_norm = hypot(goal_dx, goal_dy)
        if move_norm <= 1e-6 or goal_norm <= 1e-6:
            return 0.0

        direction_similarity = ((move_dx * goal_dx) + (move_dy * goal_dy)) / (move_norm * goal_norm)
        toward_goal = move_dx > 0.0 if attack_direction_right else move_dx < 0.0
        goal_progress = dist_start - dist_end
        receiver_is_opponent = receiver_team_id in {"home", "away"} and receiver_team_id != team_id
        receiver_is_goalkeeper = str(event.get("_receiver_class_name") or "").strip() == "goalkeeper"
        central_lane = abs(start_y - goal_y) <= SHOT_CENTRAL_LANE_DISTANCE_M

        score = 0.0
        if dist_start <= 28.0:
            score += 2.0
        elif dist_start <= 35.0:
            score += 1.0
        if toward_goal:
            score += 1.0
        if direction_similarity >= 0.97:
            score += 2.0
        elif direction_similarity >= SHOT_MIN_DIRECTION_SIMILARITY:
            score += 1.0
        if goal_progress >= 8.0:
            score += 2.0
        elif goal_progress >= SHOT_MIN_GOAL_PROGRESS_M:
            score += 1.0
        if receiver_is_goalkeeper:
            score += 2.0
        elif receiver_is_opponent:
            score += 1.0
        if central_lane:
            score += 1.0
        return score

    @staticmethod
    def _is_cross_team_recovery(event: dict[str, Any]) -> bool:
        team_id = event.get("_player_team_id")
        receiver_team_id = event.get("_receiver_team_id")
        receiver_id = _clean_id(event.get("receiver_id"))
        return (
            team_id in {"home", "away"}
            and receiver_team_id in {"home", "away"}
            and team_id != receiver_team_id
            and receiver_id is not None
        )

    def _is_control_recovery(self, event: dict[str, Any]) -> bool:
        player_id = _clean_id(event.get("player_id"))
        team_id = event.get("_player_team_id")
        previous_team_id = self._possession.team_id
        previous_player_id = self._possession.player_id
        if player_id is None or team_id not in {"home", "away"}:
            return False
        if previous_player_id is None or previous_team_id not in {"home", "away"}:
            return False
        return player_id != previous_player_id and team_id != previous_team_id

    def _apply_shot(self, event: dict[str, Any], *, shot_score: float) -> dict[str, Any]:
        event["is_pred_shot"] = True
        event["shot_direction_similarity"] = self._direction_similarity(
            event,
            attack_direction_right=bool(event.get("attack_direction_right")),
        )
        return self._finalize_event(
            event,
            semantic_type="shot",
            semantic_source="online_shot_rule",
            confidence=min(1.0, max(0.6, shot_score / 8.0)),
        )

    def _apply_robbery_from_pass(self, event: dict[str, Any]) -> dict[str, Any]:
        victim_player_id = _clean_id(event.get("player_id"))
        victim_track_id = _clean_id(event.get("player_track_id"))
        receiver_id = _clean_id(event.get("receiver_id"))
        receiver_track_id = _clean_id(event.get("receiver_track_id"))
        receiver_payload = event.get("_receiver_payload") or {}
        event["player_id"] = receiver_id
        event["receiver_id"] = victim_player_id
        event["canonical_src"] = receiver_id
        event["canonical_dst"] = victim_player_id
        event["player_track_id"] = receiver_track_id
        event["receiver_track_id"] = victim_track_id
        end_pos = _field_position(receiver_payload)
        if end_pos is not None:
            event["start_x"] = end_pos[0]
            event["start_y"] = end_pos[1]
        event["end_x"] = None
        event["end_y"] = None
        event["attack_direction_right"] = self._team_attacks_right(team_id_from_player_id(receiver_id))
        return self._finalize_event(
            event,
            semantic_type="robo",
            semantic_source="online_possession_rule",
            confidence=1.0,
        )

    def _apply_robbery_from_control(self, event: dict[str, Any]) -> dict[str, Any]:
        event["receiver_id"] = self._possession.player_id
        event["canonical_dst"] = self._possession.player_id
        event["receiver_track_id"] = self._possession.track_id
        event["end_x"] = None
        event["end_y"] = None
        return self._finalize_event(
            event,
            semantic_type="robo",
            semantic_source="online_possession_rule",
            confidence=1.0,
        )

    @staticmethod
    def _direction_similarity(event: dict[str, Any], *, attack_direction_right: bool) -> float | None:
        start_x = _safe_float(event.get("start_x"))
        start_y = _safe_float(event.get("start_y"))
        end_x = _safe_float(event.get("end_x"))
        end_y = _safe_float(event.get("end_y"))
        if None in {start_x, start_y, end_x, end_y}:
            return None
        goal_x = PITCH_LENGTH_M if attack_direction_right else 0.0
        goal_y = PITCH_CENTER_Y
        move_dx = end_x - start_x
        move_dy = end_y - start_y
        goal_dx = goal_x - start_x
        goal_dy = goal_y - start_y
        move_norm = hypot(move_dx, move_dy)
        goal_norm = hypot(goal_dx, goal_dy)
        if move_norm <= 1e-6 or goal_norm <= 1e-6:
            return None
        return ((move_dx * goal_dx) + (move_dy * goal_dy)) / (move_norm * goal_norm)

    def _update_possession(self, event: dict[str, Any]) -> None:
        semantic_type = str(event.get("event_type_semantic") or event.get("event_type") or "").strip()
        player_id = _clean_id(event.get("player_id"))
        player_track_id = _clean_id(event.get("player_track_id"))
        receiver_id = _clean_id(event.get("receiver_id"))
        receiver_track_id = _clean_id(event.get("receiver_track_id"))
        player_team_id = team_id_from_player_id(player_id)
        receiver_team_id = team_id_from_player_id(receiver_id)

        if semantic_type == "out":
            self._possession = PossessionState()
            return
        if semantic_type == "kick":
            next_player_id = receiver_id or player_id
            next_track_id = receiver_track_id or player_track_id
            next_team_id = receiver_team_id or player_team_id
            self._possession = PossessionState(next_player_id, next_track_id, next_team_id)
            return
        if semantic_type == "shot":
            next_player_id = receiver_id or player_id
            next_track_id = receiver_track_id or player_track_id
            next_team_id = receiver_team_id or player_team_id
            self._possession = PossessionState(next_player_id, next_track_id, next_team_id)
            return
        if semantic_type in {"control", "robo"}:
            self._possession = PossessionState(player_id, player_track_id, player_team_id)

    @staticmethod
    def _finalize_event(
        event: dict[str, Any],
        *,
        semantic_type: str,
        semantic_source: str,
        confidence: float,
    ) -> dict[str, Any]:
        result = dict(event)
        result["event_type_semantic"] = semantic_type
        result["event_type"] = semantic_type
        result["semantic_source"] = semantic_source
        result["semantic_confidence"] = float(confidence)
        result.pop("_player_payload", None)
        result.pop("_receiver_payload", None)
        result.pop("_player_team_id", None)
        result.pop("_receiver_team_id", None)
        result.pop("_player_team_name", None)
        result.pop("_receiver_team_name", None)
        result.pop("_receiver_class_name", None)
        return result


def _field_position(payload: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None
    raw_position = payload.get("field_position_m") or payload.get("field_position")
    if not isinstance(raw_position, (list, tuple)) or len(raw_position) < 2:
        return None
    x_val = _safe_float(raw_position[0])
    y_val = _safe_float(raw_position[1])
    if x_val is None or y_val is None:
        return None
    return (x_val, y_val)


def _event_position(event: dict[str, Any], prefix: str) -> tuple[float, float] | None:
    x_val = _safe_float(event.get(f"{prefix}_x"))
    y_val = _safe_float(event.get(f"{prefix}_y"))
    if x_val is None or y_val is None:
        return None
    return (x_val, y_val)


def _clean_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    return safe_float(value)
