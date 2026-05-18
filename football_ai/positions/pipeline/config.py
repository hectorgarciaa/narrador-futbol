from __future__ import annotations

from ..assignment import (
    DEFAULT_RATIO_PRIORITY_MIN_COUNT,
    DEFAULT_RATIO_PRIORITY_MIN_CUMULATIVE_RATIO,
    DEFAULT_RATIO_PRIORITY_MIN_FINAL_RATIO,
    normalize_expected_roles_assignment_method,
)
from .lineup_spec import normalize_slot_token


DEFAULT_SPECIAL_SEED_CANONICAL_IDS = (1, 2)
DEFAULT_SPECIAL_SEED_DEFENDER_ROLES = ("LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT")
DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH = "models/positions/20260427_002133/best_model.pt"
DEFAULT_FRAME_EXPECTED_ROLES_ASSIGNMENT_METHOD = "hungarian"
DEFAULT_SEGMENT_EXPECTED_ROLES_ASSIGNMENT_METHOD = "hungarian"
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


def resolve_frame_expected_roles_assignment_method(positions_cfg):
    return normalize_expected_roles_assignment_method(
        positions_cfg.get("frame_expected_roles_assignment_method", DEFAULT_FRAME_EXPECTED_ROLES_ASSIGNMENT_METHOD),
        default=DEFAULT_FRAME_EXPECTED_ROLES_ASSIGNMENT_METHOD,
    )


def resolve_segment_expected_roles_assignment_method(positions_cfg):
    raw_value = positions_cfg.get(
        "segment_expected_roles_assignment_method",
        positions_cfg.get(
            "role_stabilization_expected_roles_assignment",
            DEFAULT_SEGMENT_EXPECTED_ROLES_ASSIGNMENT_METHOD,
        ),
    )
    return normalize_expected_roles_assignment_method(
        raw_value,
        default=DEFAULT_SEGMENT_EXPECTED_ROLES_ASSIGNMENT_METHOD,
    )


def resolve_frame_ratio_priority_thresholds(positions_cfg):
    min_count = max(1, int(positions_cfg.get("frame_ratio_priority_min_count", 1)))
    min_cumulative_ratio = float(
        positions_cfg.get(
            "frame_ratio_priority_min_cumulative_ratio",
            DEFAULT_RATIO_PRIORITY_MIN_CUMULATIVE_RATIO,
        )
    )
    min_final_ratio = float(
        positions_cfg.get(
            "frame_ratio_priority_min_final_ratio",
            min_cumulative_ratio,
        )
    )
    return {
        "min_count": int(min_count),
        "min_cumulative_ratio": float(min_cumulative_ratio),
        "min_final_ratio": float(min_final_ratio),
    }


def resolve_segment_ratio_priority_thresholds(positions_cfg):
    min_count = max(
        1,
        int(
            positions_cfg.get(
                "segment_ratio_priority_min_count",
                positions_cfg.get(
                    "role_stabilization_expected_roles_min_count",
                    DEFAULT_RATIO_PRIORITY_MIN_COUNT,
                ),
            )
        ),
    )
    min_cumulative_ratio = float(
        positions_cfg.get(
            "segment_ratio_priority_min_cumulative_ratio",
            positions_cfg.get(
                "role_stabilization_expected_roles_min_ratio",
                DEFAULT_RATIO_PRIORITY_MIN_CUMULATIVE_RATIO,
            ),
        )
    )
    min_final_ratio = float(
        positions_cfg.get(
            "segment_ratio_priority_min_final_ratio",
            positions_cfg.get(
                "role_stabilization_expected_roles_min_final_ratio",
                min_cumulative_ratio if min_cumulative_ratio is not None else DEFAULT_RATIO_PRIORITY_MIN_FINAL_RATIO,
            ),
        )
    )
    return {
        "min_count": int(min_count),
        "min_cumulative_ratio": float(min_cumulative_ratio),
        "min_final_ratio": float(min_final_ratio),
    }
