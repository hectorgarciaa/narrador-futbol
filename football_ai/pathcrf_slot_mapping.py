from __future__ import annotations

from typing import Any


HOME_GOALKEEPER_CANONICAL_ID = 1
AWAY_GOALKEEPER_CANONICAL_ID = 2
HOME_FIELD_PLAYER_CANONICAL_IDS = tuple(range(3, 13))
AWAY_FIELD_PLAYER_CANONICAL_IDS = tuple(range(13, 23))
REFEREE_CANONICAL_IDS = (23, 24, 25)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_id_to_person_slot(canonical_id: Any) -> str | None:
    canonical_value = _safe_int(canonical_id)
    if canonical_value is None:
        return None
    if canonical_value == HOME_GOALKEEPER_CANONICAL_ID:
        return "home_1"
    if canonical_value == AWAY_GOALKEEPER_CANONICAL_ID:
        return "away_1"
    if canonical_value in HOME_FIELD_PLAYER_CANONICAL_IDS:
        return f"home_{canonical_value - 1}"
    if canonical_value in AWAY_FIELD_PLAYER_CANONICAL_IDS:
        return f"away_{canonical_value - 11}"
    return None


def canonical_id_to_referee_slot(canonical_id: Any) -> str | None:
    canonical_value = _safe_int(canonical_id)
    if canonical_value is None:
        return None
    for slot_index, referee_canonical_id in enumerate(REFEREE_CANONICAL_IDS, start=1):
        if canonical_value == referee_canonical_id:
            return f"referee_{slot_index}"
    return None


def person_slot_to_canonical_id(slot_name: Any) -> int | None:
    text = str(slot_name or "").strip().lower()
    if text == "home_1":
        return HOME_GOALKEEPER_CANONICAL_ID
    if text == "away_1":
        return AWAY_GOALKEEPER_CANONICAL_ID
    if text.startswith("home_"):
        slot_number = _safe_int(text.split("_", maxsplit=1)[1])
        if slot_number is not None and 2 <= slot_number <= 11:
            return slot_number + 1
    if text.startswith("away_"):
        slot_number = _safe_int(text.split("_", maxsplit=1)[1])
        if slot_number is not None and 2 <= slot_number <= 11:
            return slot_number + 11
    return None


def referee_slot_to_canonical_id(slot_name: Any) -> int | None:
    text = str(slot_name or "").strip().lower()
    if not text.startswith("referee_"):
        return None
    slot_number = _safe_int(text.split("_", maxsplit=1)[1])
    if slot_number is None or slot_number <= 0 or slot_number > len(REFEREE_CANONICAL_IDS):
        return None
    return REFEREE_CANONICAL_IDS[slot_number - 1]


def all_person_slots(expected_players_per_team: int = 11) -> list[str]:
    return [f"home_{idx}" for idx in range(1, expected_players_per_team + 1)] + [
        f"away_{idx}" for idx in range(1, expected_players_per_team + 1)
    ]


def all_referee_slots(expected_referees: int = 3) -> list[str]:
    return [f"referee_{idx}" for idx in range(1, expected_referees + 1)]
