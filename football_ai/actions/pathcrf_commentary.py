from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.core.serialization import convert_to_serializable

from .pathcrf_semantics import (
    PITCH_LENGTH_M,
    ensure_semantic_event_columns,
    infer_team_attack_directions,
    is_player_slot,
    parse_timestamp_seconds,
    safe_float,
    sort_events,
    team_attacks_right,
    team_id_from_player_id,
)


COMMENTARY_ACTION_MAP = {
    "control": "control",
    "kick": "pase",
    "robo": "robo",
    "shot": "tiro",
    "corner": "corner",
    "throw_in": "fuera de banda",
    "goalkick": "saque de puerta",
}
TEAM_IN_FAVOR_ACTIONS = {"corner", "fuera de banda", "saque de puerta"}
SKIP_EVENT_TYPES = {"out"}
FIELD_ZONE_LABELS = {
    "iniciacion": "zona de iniciacion",
    "creacion": "zona de creacion",
    "finalizacion": "zona de finalizacion",
}


@dataclass
class TrackIdentity:
    track_id: str
    class_name: str
    team_name: str | None
    player_name: str | None
    player_position: str | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _best_value(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return max(counter.items(), key=lambda item: (int(item[1]), str(item[0])))[0]


def _normalize_position(payload: dict[str, Any]) -> str | None:
    for key in ("lineup_slot", "predicted_role", "predicted_role_frame"):
        value = _clean_text(payload.get(key))
        if value:
            return value
    return None


def _build_track_identity_maps(tracks: dict[str, Any]) -> tuple[dict[str, TrackIdentity], dict[int, dict[str, dict[str, Any]]]]:
    counters: dict[str, dict[str, Counter[str]]] = {}
    class_by_track: dict[str, str] = {}
    frame_index: dict[int, dict[str, dict[str, Any]]] = {}

    for class_name in ("player", "goalkeeper"):
        for frame_id, frame_map in enumerate(tracks.get(class_name, []) or []):
            if not isinstance(frame_map, dict):
                continue
            frame_tracks = frame_index.setdefault(int(frame_id), {})
            for raw_track_id, raw_payload in frame_map.items():
                if not isinstance(raw_payload, dict):
                    continue
                track_id = str(raw_track_id)
                frame_tracks[track_id] = dict(raw_payload)
                class_by_track.setdefault(track_id, class_name)
                bucket = counters.setdefault(
                    track_id,
                    {
                        "team": Counter(),
                        "player_name": Counter(),
                        "player_position": Counter(),
                    },
                )
                team_name = _clean_text(raw_payload.get("team"))
                if team_name:
                    bucket["team"][team_name] += 1
                player_name = _clean_text(raw_payload.get("player_name"))
                if player_name:
                    bucket["player_name"][player_name] += 1
                player_position = _normalize_position(raw_payload)
                if player_position:
                    bucket["player_position"][player_position] += 1

    identities = {
        track_id: TrackIdentity(
            track_id=track_id,
            class_name=class_by_track.get(track_id, "player"),
            team_name=_best_value(bucket["team"]),
            player_name=_best_value(bucket["player_name"]),
            player_position=_best_value(bucket["player_position"]),
        )
        for track_id, bucket in counters.items()
    }
    return identities, frame_index


def _load_tracks(tracks_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(tracks_path).expanduser().resolve()
    with resolved_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slot_mapping_from_summary(conversion_summary: dict[str, Any] | None) -> tuple[dict[str, str], set[str]]:
    if not conversion_summary:
        return {}, set()

    raw_to_slot = dict(conversion_summary.get("person_slot_assignments") or {})
    slot_to_raw = {str(slot): str(raw_id) for raw_id, slot in raw_to_slot.items()}

    raw_ref_to_slot = dict(conversion_summary.get("referee_slot_assignments") or {})
    slot_to_raw.update({str(slot): str(raw_id) for raw_id, slot in raw_ref_to_slot.items()})

    synthetic_slots = {
        str(value)
        for key in ("synthetic_home_slots", "synthetic_away_slots", "synthetic_referee_slots")
        for value in (conversion_summary.get(key) or [])
    }
    return slot_to_raw, synthetic_slots


def _fallback_player_name(track_id: str | None) -> str | None:
    if not track_id:
        return None
    return f"jugador {track_id}"


def _fallback_player_position(identity: TrackIdentity | None, payload: dict[str, Any] | None = None) -> str:
    if payload is not None:
        position = _normalize_position(payload)
        if position:
            return position
    if identity is not None and identity.player_position:
        return str(identity.player_position)
    return "JUG"


def _resolve_track_payload(
    frame_index: dict[int, dict[str, dict[str, Any]]],
    frame_id: int | None,
    track_id: str | None,
) -> dict[str, Any] | None:
    if frame_id is None or track_id is None:
        return None
    return dict(frame_index.get(int(frame_id), {}).get(str(track_id), {})) or None


def _field_position_from_payload(payload: dict[str, Any] | None) -> tuple[float, float] | None:
    if not payload:
        return None
    raw_position = payload.get("field_position_m") or payload.get("field_position")
    if raw_position is None:
        return None
    try:
        values = list(raw_position)
        x_m = float(values[0])
        y_m = float(values[1])
    except (TypeError, ValueError, IndexError):
        return None
    if safe_float(x_m) is None or safe_float(y_m) is None:
        return None
    return x_m, y_m


def _field_position_from_event_row(row: Any) -> tuple[float, float] | None:
    x_m = safe_float(getattr(row, "start_x", None))
    y_m = safe_float(getattr(row, "start_y", None))
    if x_m is None or y_m is None:
        return None
    return x_m, y_m


def _field_zone_key(
    field_position_m: tuple[float, float] | None,
    *,
    attacks_right: bool,
) -> str | None:
    if field_position_m is None:
        return None
    x_m = max(0.0, min(PITCH_LENGTH_M, float(field_position_m[0])))
    first_third = PITCH_LENGTH_M / 3.0
    second_third = (2.0 * PITCH_LENGTH_M) / 3.0
    if attacks_right:
        if x_m < first_third:
            return "iniciacion"
        if x_m < second_third:
            return "creacion"
        return "finalizacion"
    if x_m < first_third:
        return "finalizacion"
    if x_m < second_third:
        return "creacion"
    return "iniciacion"


def _commentary_priority_for_zone(action: str | None, field_zone_key: str | None) -> str:
    if action in {"tiro", "gol"} or field_zone_key == "finalizacion":
        return "high"
    if field_zone_key == "iniciacion":
        return "low"
    return "normal"


def _infer_team_names(identities: dict[str, TrackIdentity]) -> list[str]:
    teams = sorted(
        {
            identity.team_name
            for identity in identities.values()
            if identity.team_name
        }
    )
    return teams


def _opponent_team(team_name: str | None, all_team_names: list[str]) -> str | None:
    if team_name is None:
        return None
    for candidate in all_team_names:
        if candidate != team_name:
            return candidate
    return None


def _commentary_action_for_event(event_type: Any) -> str | None:
    normalized = _clean_text(event_type)
    if not normalized:
        return None
    return COMMENTARY_ACTION_MAP.get(normalized)


def _same_team_name(left: str | None, right: str | None) -> bool:
    return bool(left and right and left.casefold() == right.casefold())


def build_commentary_events_json(
    *,
    events: pd.DataFrame,
    tracks_path: str | Path,
    conversion_summary: dict[str, Any] | None,
    output_path: str | Path,
    source_paths: dict[str, Any] | None = None,
) -> Path:
    events_df = ensure_semantic_event_columns(sort_events(events))
    attack_directions = infer_team_attack_directions(events_df)
    tracks = _load_tracks(tracks_path)
    identities, frame_index = _build_track_identity_maps(tracks)
    slot_to_track_id, synthetic_slots = _slot_mapping_from_summary(conversion_summary)
    all_team_names = _infer_team_names(identities)

    enriched_events: list[dict[str, Any]] = []
    ready_commentary_events: list[dict[str, Any]] = []
    skipped_reasons = Counter()
    current_possession_slot: str | None = None
    current_possession_track_id: str | None = None
    current_possession_team_name: str | None = None

    for action_index, row in enumerate(events_df.itertuples(index=False), start=1):
        raw_event_type = _clean_text(getattr(row, "event_type_raw", None))
        semantic_event_type = _clean_text(getattr(row, "event_type_semantic", None)) or raw_event_type
        commentary_action = _commentary_action_for_event(semantic_event_type)
        frame_id = int(row.frame_id) if safe_float(getattr(row, "frame_id", None)) is not None else None
        event_time_s = parse_timestamp_seconds(getattr(row, "timestamp", None))

        pathcrf_player_id = _clean_text(getattr(row, "player_id", None))
        pathcrf_receiver_id = _clean_text(getattr(row, "receiver_id", None))
        is_synthetic_slot = pathcrf_player_id in synthetic_slots if pathcrf_player_id else False

        track_id = slot_to_track_id.get(pathcrf_player_id) if pathcrf_player_id else None
        receiver_track_id = slot_to_track_id.get(pathcrf_receiver_id) if pathcrf_receiver_id else None

        identity = identities.get(str(track_id)) if track_id is not None else None
        receiver_identity = identities.get(str(receiver_track_id)) if receiver_track_id is not None else None

        track_payload = _resolve_track_payload(frame_index, frame_id, track_id)
        receiver_payload = _resolve_track_payload(frame_index, frame_id, receiver_track_id)

        team_name = _clean_text((track_payload or {}).get("team")) or (identity.team_name if identity else None)
        opponent_team_name = _opponent_team(team_name, all_team_names)
        player_name = (
            _clean_text((track_payload or {}).get("player_name"))
            or (identity.player_name if identity else None)
            or _fallback_player_name(track_id)
        )
        player_position = _fallback_player_position(identity, track_payload)
        action_target = (
            _clean_text((receiver_payload or {}).get("player_name"))
            or (receiver_identity.player_name if receiver_identity else None)
        )
        receiver_team_name = _clean_text((receiver_payload or {}).get("team")) or (
            receiver_identity.team_name if receiver_identity else None
        )
        team_in_favor = team_name if commentary_action in TEAM_IN_FAVOR_ACTIONS else None
        semantic_source = _clean_text(getattr(row, "semantic_source", None))
        semantic_confidence = getattr(row, "semantic_confidence", None)
        possession_postprocess: str | None = None

        original_pathcrf_player_id = pathcrf_player_id
        original_pathcrf_receiver_id = pathcrf_receiver_id
        original_track_id = track_id
        original_receiver_track_id = receiver_track_id
        original_player_name = player_name
        original_receiver_name = action_target

        commentary_ready = True
        skip_reason = None
        if (
            semantic_event_type == "kick"
            and pathcrf_receiver_id
            and is_player_slot(pathcrf_receiver_id)
            and team_name
            and receiver_team_name
            and not _same_team_name(team_name, receiver_team_name)
        ):
            pathcrf_player_id = original_pathcrf_receiver_id
            pathcrf_receiver_id = original_pathcrf_player_id
            track_id = original_receiver_track_id
            receiver_track_id = original_track_id
            identity = receiver_identity
            track_payload = receiver_payload
            team_name = receiver_team_name
            opponent_team_name = _opponent_team(team_name, all_team_names)
            player_name = original_receiver_name or _fallback_player_name(track_id)
            player_position = _fallback_player_position(identity, track_payload)
            action_target = original_player_name
            semantic_event_type = "robo"
            commentary_action = "robo"
            team_in_favor = None
            possession_postprocess = "cross_team_pass_to_robbery"
            semantic_source = "possession_rule"
        elif (
            semantic_event_type == "kick"
            and current_possession_slot is not None
            and pathcrf_player_id != current_possession_slot
        ):
            commentary_ready = False
            skip_reason = "possession_actor_mismatch"
            possession_postprocess = "skip_kick_from_non_possessor"
        elif (
            semantic_event_type == "control"
            and current_possession_slot is not None
            and pathcrf_player_id != current_possession_slot
        ):
            if (
                current_possession_team_name
                and team_name
                and not _same_team_name(current_possession_team_name, team_name)
            ):
                semantic_event_type = "robo"
                commentary_action = "robo"
                team_in_favor = None
                previous_identity = (
                    identities.get(str(current_possession_track_id))
                    if current_possession_track_id is not None
                    else None
                )
                action_target = (
                    previous_identity.player_name
                    if previous_identity is not None and previous_identity.player_name
                    else _fallback_player_name(current_possession_track_id)
                )
                possession_postprocess = "control_after_opponent_possession_to_robbery"
                semantic_source = "possession_rule"
            else:
                commentary_ready = False
                skip_reason = "possession_actor_mismatch"
                possession_postprocess = "skip_control_from_non_possessor"

        is_synthetic_slot = pathcrf_player_id in synthetic_slots if pathcrf_player_id else False
        if semantic_event_type in SKIP_EVENT_TYPES:
            commentary_ready = False
            skip_reason = "unsupported_event_type"
        elif not pathcrf_player_id or not is_player_slot(pathcrf_player_id):
            commentary_ready = False
            skip_reason = "non_player_actor"
        elif is_synthetic_slot or track_id is None:
            commentary_ready = False
            skip_reason = "missing_real_track_id"
        elif commentary_action is None:
            commentary_ready = False
            skip_reason = "unmapped_event_type"
        elif event_time_s is None:
            commentary_ready = False
            skip_reason = "invalid_timestamp"

        commentary_event = None
        if commentary_ready:
            pathcrf_team_id = team_id_from_player_id(pathcrf_player_id)
            period_id = getattr(row, "period_id", None)
            attacks_right = team_attacks_right(
                period_id,
                pathcrf_team_id,
                attack_directions,
            )
            field_position_m = (
                _field_position_from_payload(track_payload)
                or _field_position_from_event_row(row)
            )
            field_zone_key = _field_zone_key(
                field_position_m,
                attacks_right=attacks_right,
            )
            field_zone = FIELD_ZONE_LABELS.get(field_zone_key)
            commentary_priority = _commentary_priority_for_zone(
                commentary_action,
                field_zone_key,
            )
            commentary_event = {
                "action": commentary_action,
                "event_time_s": float(event_time_s),
                "player_name": str(player_name),
                "player_position": str(player_position),
                "team_name": team_name,
                "opponent_team_name": opponent_team_name,
                "team_in_favor": team_in_favor,
                "field_zone": field_zone,
                "action_target": action_target,
                "action_index": int(action_index),
            }
            ready_commentary_events.append(
                {
                    "event": commentary_event,
                    "metadata": {
                        "action_index": int(action_index),
                        "frame_id": frame_id,
                        "episode_id": getattr(row, "episode_id", None),
                        "pathcrf_player_id": pathcrf_player_id,
                        "pathcrf_receiver_id": pathcrf_receiver_id,
                        "track_id": int(track_id) if str(track_id).isdigit() else track_id,
                        "receiver_track_id": (
                            int(receiver_track_id)
                            if receiver_track_id is not None and str(receiver_track_id).isdigit()
                            else receiver_track_id
                        ),
                        "event_type_raw": raw_event_type,
                        "event_type_semantic": semantic_event_type,
                        "semantic_source": semantic_source,
                        "possession_postprocess": possession_postprocess,
                        "field_position_m": (
                            [float(field_position_m[0]), float(field_position_m[1])]
                            if field_position_m is not None
                            else None
                        ),
                        "field_zone_key": field_zone_key,
                        "field_zone": field_zone,
                        "attack_direction_right": bool(attacks_right),
                        "pathcrf_team_id": pathcrf_team_id,
                        "commentary_priority": commentary_priority,
                    },
                }
            )
            if (
                commentary_action == "pase"
                and pathcrf_receiver_id
                and is_player_slot(pathcrf_receiver_id)
            ):
                current_possession_slot = pathcrf_receiver_id
                current_possession_track_id = receiver_track_id
                current_possession_team_name = receiver_team_name or team_name
            else:
                current_possession_slot = pathcrf_player_id
                current_possession_track_id = track_id
                current_possession_team_name = team_name
        else:
            field_position_m = (
                _field_position_from_payload(track_payload)
                or _field_position_from_event_row(row)
            )
            field_zone_key = None
            field_zone = None
            commentary_priority = None
            attacks_right = None
            pathcrf_team_id = team_id_from_player_id(pathcrf_player_id)
            skipped_reasons[str(skip_reason or "unknown")] += 1

        enriched_events.append(
            {
                "action_index": int(action_index),
                "frame_id": frame_id,
                "episode_id": getattr(row, "episode_id", None),
                "period_id": getattr(row, "period_id", None),
                "timestamp": getattr(row, "timestamp", None),
                "event_time_s": event_time_s,
                "pathcrf_player_id": pathcrf_player_id,
                "pathcrf_receiver_id": pathcrf_receiver_id,
                "track_id": int(track_id) if track_id is not None and str(track_id).isdigit() else track_id,
                "receiver_track_id": (
                    int(receiver_track_id)
                    if receiver_track_id is not None and str(receiver_track_id).isdigit()
                    else receiver_track_id
                ),
                "event_type_raw": raw_event_type,
                "event_type_semantic": semantic_event_type,
                "semantic_source": semantic_source,
                "semantic_confidence": semantic_confidence,
                "possession_postprocess": possession_postprocess,
                "commentary_action": commentary_action,
                "commentary_ready": commentary_ready,
                "skip_reason": skip_reason,
                "is_synthetic_slot": bool(is_synthetic_slot),
                "possession_slot_after": current_possession_slot,
                "possession_track_id_after": (
                    int(current_possession_track_id)
                    if current_possession_track_id is not None and str(current_possession_track_id).isdigit()
                    else current_possession_track_id
                ),
                "possession_team_after": current_possession_team_name,
                "team_name": team_name,
                "opponent_team_name": opponent_team_name,
                "player_name": player_name,
                "player_position": player_position,
                "receiver_name": action_target,
                "start_x": getattr(row, "start_x", None),
                "start_y": getattr(row, "start_y", None),
                "end_x": getattr(row, "end_x", None),
                "end_y": getattr(row, "end_y", None),
                "field_position_m": (
                    [float(field_position_m[0]), float(field_position_m[1])]
                    if field_position_m is not None
                    else None
                ),
                "field_zone_key": field_zone_key,
                "field_zone": field_zone,
                "attack_direction_right": attacks_right,
                "pathcrf_team_id": pathcrf_team_id,
                "commentary_priority": commentary_priority,
                "commentary_event": commentary_event,
            }
        )

    output_payload = {
        "generated_at_utc": _now_iso(),
        "source": {
            "tracks_path": str(Path(tracks_path).expanduser().resolve()),
            **{str(key): value for key, value in (source_paths or {}).items()},
        },
        "stats": {
            "semantic_events": int(len(events_df)),
            "commentary_ready_events": int(len(ready_commentary_events)),
            "skipped_events": int(len(events_df) - len(ready_commentary_events)),
            "skipped_by_reason": dict(skipped_reasons),
        },
        "events": enriched_events,
        "commentary_events": ready_commentary_events,
    }

    target_path = Path(output_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(output_payload), f, indent=2, ensure_ascii=False)
    return target_path
