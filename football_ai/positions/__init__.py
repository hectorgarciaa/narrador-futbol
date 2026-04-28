"""
Positional role helpers and online assignment logic.
"""

from .pipeline import (
    LineupSlotMatcher,
    LineupSpecError,
    OnlineSpecialSeedRoleAssigner,
    PositionInferingPhase,
    base_role_token,
    build_expected_roles_by_team,
    build_role_artifacts_output_dir,
    build_role_predictions_output_paths,
    build_team_colors_by_team,
    copy_output_artifact,
    get_formation_catalog,
    load_lineup_spec,
    normalize_slot_token,
    save_dataframe_csv,
    sanitize_video_stem,
    validate_lineup_payload,
)

__all__ = [
    "LineupSlotMatcher",
    "LineupSpecError",
    "OnlineSpecialSeedRoleAssigner",
    "PositionInferingPhase",
    "base_role_token",
    "build_role_artifacts_output_dir",
    "build_role_predictions_output_paths",
    "build_expected_roles_by_team",
    "build_team_colors_by_team",
    "copy_output_artifact",
    "get_formation_catalog",
    "load_lineup_spec",
    "normalize_slot_token",
    "save_dataframe_csv",
    "sanitize_video_stem",
    "validate_lineup_payload",
]
