from __future__ import annotations

from .lineup_spec import normalize_slot_token


DEFAULT_SPECIAL_SEED_CANONICAL_IDS = (1, 2)
DEFAULT_SPECIAL_SEED_DEFENDER_ROLES = ("LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT")
DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH = "models/positions/20260427_002133/best_model.pt"
DEFAULT_EXPECTED_ROLES_BY_TEAM = {
    "Real Madrid": ["POR", "LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT", "MC", "MI", "MD", "DC", "DC"],
    "Wolfsburgo": ["POR", "LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT", "MC", "MI", "MD", "DC", "DC"],
}
SEGMENT_POSITION_WEIGHT = 1.25
SEGMENT_CONFIDENCE_WEIGHT = 0.35


def normalize_expected_roles_mapping(expected_roles_by_team):
    if not expected_roles_by_team:
        return {}
    return {
        str(team_name): [normalize_slot_token(role) for role in list(roles or []) if normalize_slot_token(role)]
        for team_name, roles in dict(expected_roles_by_team).items()
    }


def resolve_expected_roles_by_team_from_config(config):
    positions_cfg = getattr(config, "positions", {}) or {}
    configured = positions_cfg.get("expected_roles_by_team")
    return normalize_expected_roles_mapping(configured or DEFAULT_EXPECTED_ROLES_BY_TEAM)
