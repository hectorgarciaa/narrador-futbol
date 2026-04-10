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
    ensure_semantic_event_columns,
    is_player_slot,
    parse_timestamp_seconds,
    safe_float,
    sort_events,
)


COMMENTARY_ACTION_MAP = {
    "control": "control",
    "kick": "pase",
    "shot": "tiro",
    "corner": "corner",
    "throw_in": "fuera de banda",
    "goalkick": "saque de puerta",
}
TEAM_IN_FAVOR_ACTIONS = {"corner", "fuera de banda", "saque de puerta"}
SKIP_EVENT_TYPES = {"out"}


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


def build_commentary_events_json(
    *,
    events: pd.DataFrame,
    tracks_path: str | Path,
    conversion_summary: dict[str, Any] | None,
    output_path: str | Path,
    source_paths: dict[str, Any] | None = None,
) -> Path:
    events_df = ensure_semantic_event_columns(sort_events(events))
    tracks = _load_tracks(tracks_path)
    identities, frame_index = _build_track_identity_maps(tracks)
    slot_to_track_id, synthetic_slots = _slot_mapping_from_summary(conversion_summary)
    all_team_names = _infer_team_names(identities)

    enriched_events: list[dict[str, Any]] = []
    ready_commentary_events: list[dict[str, Any]] = []
    skipped_reasons = Counter()

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
        team_in_favor = team_name if commentary_action in TEAM_IN_FAVOR_ACTIONS else None

        commentary_ready = True
        skip_reason = None
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
            commentary_event = {
                "action": commentary_action,
                "event_time_s": float(event_time_s),
                "player_name": str(player_name),
                "player_position": str(player_position),
                "team_name": team_name,
                "opponent_team_name": opponent_team_name,
                "team_in_favor": team_in_favor,
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
                        "semantic_source": _clean_text(getattr(row, "semantic_source", None)),
                    },
                }
            )
        else:
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
                "semantic_source": _clean_text(getattr(row, "semantic_source", None)),
                "semantic_confidence": getattr(row, "semantic_confidence", None),
                "commentary_action": commentary_action,
                "commentary_ready": commentary_ready,
                "skip_reason": skip_reason,
                "is_synthetic_slot": bool(is_synthetic_slot),
                "team_name": team_name,
                "opponent_team_name": opponent_team_name,
                "player_name": player_name,
                "player_position": player_position,
                "receiver_name": action_target,
                "start_x": getattr(row, "start_x", None),
                "start_y": getattr(row, "start_y", None),
                "end_x": getattr(row, "end_x", None),
                "end_y": getattr(row, "end_y", None),
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
