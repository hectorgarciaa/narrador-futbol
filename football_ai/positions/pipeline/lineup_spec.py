"""
Especificaciones de alineación introducidas por interfaz y helpers de matching.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


FORMATION_CATALOG = {
    "4-3-3": {
        "ui_slots": [
            "POR",
            "LD",
            "LI",
            "DFC_DER",
            "DFC_IZQ",
            "MC",
            "MI",
            "MD",
            "DC",
            "ED",
            "EI",
        ],
        "tracking_slots": [
            "POR",
            "LD",
            "LI",
            "DFC_DER",
            "DFC_IZQ",
            "MC",
            "MI",
            "MD",
            "DC",
            "ED",
            "EI",
        ],
        "pitch_layout": {
            "POR": {"x": 50, "y": 90},
            "LI": {"x": 18, "y": 72},
            "DFC_IZQ": {"x": 38, "y": 72},
            "DFC_DER": {"x": 62, "y": 72},
            "LD": {"x": 82, "y": 72},
            "MI": {"x": 28, "y": 48},
            "MC": {"x": 50, "y": 44},
            "MD": {"x": 72, "y": 48},
            "EI": {"x": 20, "y": 20},
            "DC": {"x": 50, "y": 14},
            "ED": {"x": 80, "y": 20},
        },
    },
    "5-3-2": {
        "ui_slots": [
            "POR",
            "LD",
            "LI",
            "DFC_DER",
            "DFC_CENT",
            "DFC_IZQ",
            "MC",
            "MI",
            "MD",
            "DC_IZQ",
            "DC_DCHO",
        ],
        "tracking_slots": [
            "POR",
            "LD",
            "LI",
            "DFC_DER",
            "DFC_CENT",
            "DFC_IZQ",
            "MC",
            "MI",
            "MD",
            "DC",
            "DC",
        ],
        "pitch_layout": {
            "POR": {"x": 50, "y": 90},
            "LI": {"x": 12, "y": 62},
            "DFC_IZQ": {"x": 32, "y": 72},
            "DFC_CENT": {"x": 50, "y": 74},
            "DFC_DER": {"x": 68, "y": 72},
            "LD": {"x": 88, "y": 62},
            "MI": {"x": 28, "y": 42},
            "MC": {"x": 50, "y": 38},
            "MD": {"x": 72, "y": 42},
            "DC_IZQ": {"x": 40, "y": 16},
            "DC_DCHO": {"x": 60, "y": 16},
        },
    },
    "4-4-2": {
        "ui_slots": [
            "POR",
            "LD",
            "LI",
            "DFC_DER",
            "DFC_IZQ",
            "MC_IZQ",
            "MC_DCHO",
            "MI",
            "MD",
            "DC_IZQ",
            "DC_DCHO",
        ],
        "tracking_slots": [
            "POR",
            "LD",
            "LI",
            "DFC_DER",
            "DFC_IZQ",
            "MC",
            "MC",
            "MI",
            "MD",
            "DC",
            "DC",
        ],
        "pitch_layout": {
            "POR": {"x": 50, "y": 90},
            "LI": {"x": 18, "y": 72},
            "DFC_IZQ": {"x": 38, "y": 72},
            "DFC_DER": {"x": 62, "y": 72},
            "LD": {"x": 82, "y": 72},
            "MI": {"x": 14, "y": 42},
            "MC_IZQ": {"x": 38, "y": 46},
            "MC_DCHO": {"x": 62, "y": 46},
            "MD": {"x": 86, "y": 42},
            "DC_IZQ": {"x": 40, "y": 18},
            "DC_DCHO": {"x": 60, "y": 18},
        },
    },
}


class LineupSpecError(ValueError):
    """Error de validación para specs de alineaciones."""


def normalize_slot_token(value):
    token = str(value or "").strip().upper()
    token = token.replace("-", "_").replace(" ", "_")
    token = "_".join(part for part in token.split("_") if part)
    return token


def base_role_token(value):
    token = normalize_slot_token(value)
    if token in {"MC_IZQ", "MC_DCHO"}:
        return "MC"
    if token in {"DC_IZQ", "DC_DCHO"}:
        return "DC"
    return token


def get_formation_catalog():
    return {
        formation_name: {
            "ui_slots": list(definition["ui_slots"]),
            "tracking_slots": list(definition["tracking_slots"]),
            "pitch_layout": {
                slot: dict(coords)
                for slot, coords in definition["pitch_layout"].items()
            },
        }
        for formation_name, definition in FORMATION_CATALOG.items()
    }


def _normalize_players_by_slot(players_by_slot, formation_name):
    if not isinstance(players_by_slot, dict):
        raise LineupSpecError(
            f"`players_by_slot` debe ser un objeto/dict en la formación {formation_name}."
        )

    ui_slots = FORMATION_CATALOG[formation_name]["ui_slots"]
    expected_slots = {normalize_slot_token(slot): slot for slot in ui_slots}
    normalized = {}

    for raw_slot, raw_player_name in players_by_slot.items():
        slot = normalize_slot_token(raw_slot)
        if slot not in expected_slots:
            raise LineupSpecError(
                f"Slot no soportado '{raw_slot}' para la formación {formation_name}. "
                f"Slots válidos: {', '.join(ui_slots)}."
            )
        player_name = str(raw_player_name or "").strip()
        if not player_name:
            raise LineupSpecError(f"El slot {expected_slots[slot]} no puede ir vacío.")
        normalized[slot] = player_name

    missing = [slot for slot in expected_slots if slot not in normalized]
    extra = [slot for slot in normalized if slot not in expected_slots]
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"faltan: {', '.join(missing)}")
        if extra:
            detail.append(f"sobran: {', '.join(extra)}")
        raise LineupSpecError(
            f"La alineación {formation_name} no encaja con sus slots ({'; '.join(detail)})."
        )
    return normalized


def validate_lineup_payload(payload):
    if not isinstance(payload, dict):
        raise LineupSpecError("El spec de alineaciones debe ser un JSON objeto.")

    raw_teams = payload.get("teams")
    if not isinstance(raw_teams, list) or len(raw_teams) != 2:
        raise LineupSpecError("El spec debe contener exactamente dos equipos en `teams`.")

    video_source = str(payload.get("video_source") or "").strip()
    normalized_teams = []
    used_team_names = set()

    for idx, raw_team in enumerate(raw_teams, start=1):
        if not isinstance(raw_team, dict):
            raise LineupSpecError(f"El equipo #{idx} debe ser un objeto.")

        team_name = str(raw_team.get("team_name") or "").strip()
        team_color = str(raw_team.get("team_color") or "").strip()
        formation = str(raw_team.get("formation") or "").strip()

        if not team_name:
            raise LineupSpecError(f"El equipo #{idx} necesita `team_name`.")
        if team_name in used_team_names:
            raise LineupSpecError(f"El nombre de equipo '{team_name}' está repetido.")
        if not team_color:
            raise LineupSpecError(
                f"El equipo '{team_name}' necesita `team_color`."
            )
        if formation not in FORMATION_CATALOG:
            raise LineupSpecError(
                f"Formación no soportada '{formation}' para '{team_name}'. "
                f"Opciones: {', '.join(FORMATION_CATALOG.keys())}."
            )

        normalized_players = _normalize_players_by_slot(
            raw_team.get("players_by_slot", {}),
            formation_name=formation,
        )
        tracking_slots = list(FORMATION_CATALOG[formation]["tracking_slots"])
        ui_slots = list(FORMATION_CATALOG[formation]["ui_slots"])
        pitch_layout = {
            slot: dict(coords)
            for slot, coords in FORMATION_CATALOG[formation]["pitch_layout"].items()
        }

        normalized_teams.append(
            {
                "team_name": team_name,
                "team_color": team_color,
                "formation": formation,
                "ui_slots": ui_slots,
                "tracking_slots": tracking_slots,
                "pitch_layout": pitch_layout,
                "players_by_slot": normalized_players,
            }
        )
        used_team_names.add(team_name)

    return {
        "video_source": video_source,
        "teams": normalized_teams,
    }


def load_lineup_spec(spec_path, project_root=None):
    resolved_path = Path(spec_path).expanduser()
    if not resolved_path.is_absolute():
        base_root = Path(project_root) if project_root is not None else Path.cwd()
        resolved_path = (base_root / resolved_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"No existe el lineup spec: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    normalized = validate_lineup_payload(payload)
    normalized["spec_path"] = str(resolved_path)
    return normalized


def build_expected_roles_by_team(lineup_spec):
    normalized = validate_lineup_payload(lineup_spec)
    return {
        str(team["team_name"]): list(team["tracking_slots"])
        for team in normalized["teams"]
    }


def build_team_colors_by_team(lineup_spec):
    normalized = validate_lineup_payload(lineup_spec)
    return {
        str(team["team_name"]): str(team["team_color"])
        for team in normalized["teams"]
    }


class LineupSlotMatcher:
    """Resuelve nombre de jugador a partir de equipo + slot estabilizado."""

    def __init__(self, lineup_spec):
        normalized = validate_lineup_payload(lineup_spec)
        self.lineup_spec = normalized
        self.players_by_team = {}
        self.base_slot_counts_by_team = {}
        for team in normalized["teams"]:
            team_name = str(team["team_name"])
            players_by_slot = {
                normalize_slot_token(slot): str(player_name)
                for slot, player_name in team["players_by_slot"].items()
            }
            self.players_by_team[team_name] = players_by_slot
            self.base_slot_counts_by_team[team_name] = Counter(
                base_role_token(slot) for slot in players_by_slot.keys()
            )

    def _slot_is_unique(self, team_name, slot_name):
        base_slot = base_role_token(slot_name)
        return int(self.base_slot_counts_by_team.get(str(team_name), {}).get(base_slot, 0)) <= 1

    def resolve_slot(self, team_name, *slot_candidates):
        team_key = str(team_name or "").strip()
        players_by_slot = self.players_by_team.get(team_key)
        if not players_by_slot:
            return None

        for raw_slot in slot_candidates:
            slot = normalize_slot_token(raw_slot)
            if not slot:
                continue
            if slot in players_by_slot:
                return slot
            base_slot = base_role_token(slot)
            if base_slot in players_by_slot and self._slot_is_unique(team_key, base_slot):
                return base_slot
        return None

    def resolve_player_name(self, team_name, *slot_candidates):
        resolved_slot = self.resolve_slot(team_name, *slot_candidates)
        if resolved_slot is None:
            return None, None
        team_key = str(team_name or "").strip()
        return resolved_slot, self.players_by_team.get(team_key, {}).get(resolved_slot)
