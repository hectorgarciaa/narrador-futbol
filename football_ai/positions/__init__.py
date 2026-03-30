"""
Positional role helpers and online assignment logic.
"""

from .lineup_spec import (
    LineupSlotMatcher,
    LineupSpecError,
    base_role_token,
    build_expected_roles_by_team,
    build_team_colors_by_team,
    get_formation_catalog,
    load_lineup_spec,
    normalize_slot_token,
    validate_lineup_payload,
)
from .tracking_roles import (
    OnlineSpecialSeedRoleAssigner,
    apply_special_seed_role_team_assignment,
    build_role_artifacts_output_dir,
    build_role_predictions_output_paths,
    copy_output_artifact,
    save_dataframe_csv,
    save_role_visualizations,
    sanitize_video_stem,
)

__all__ = [
    "LineupSlotMatcher",
    "LineupSpecError",
    "OnlineSpecialSeedRoleAssigner",
    "apply_special_seed_role_team_assignment",
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
    "save_role_visualizations",
    "sanitize_video_stem",
    "validate_lineup_payload",
]
