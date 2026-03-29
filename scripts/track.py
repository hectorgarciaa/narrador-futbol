import argparse
import colorsys
import csv
import difflib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Permite ejecutar `python scripts/track.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.tracking import Tracker
from football_ai.evaluation import Evaluator
from football_ai.visualization import Drawer
from football_ai.core import get_config, get_logger, Logger, convert_to_serializable
from football_ai.positions import (
    OnlineSpecialSeedRoleAssigner,
    build_role_artifacts_output_dir,
    build_role_predictions_output_paths,
    copy_output_artifact,
    save_dataframe_csv,
    save_role_visualizations,
)

COLOR_NAME_TO_RGB = {
    "white": (255, 255, 255),
    "blanco": (255, 255, 255),
    "black": (0, 0, 0),
    "negro": (0, 0, 0),
    "red": (255, 0, 0),
    "rojo": (255, 0, 0),
    "blue": (0, 102, 255),
    "azul": (0, 102, 255),
    "light-blue": (173, 216, 230),
    "azul-claro": (173, 216, 230),
    "green": (0, 170, 0),
    "verde": (0, 170, 0),
    "light-green": (144, 238, 144),
    "verde-claro": (144, 238, 144),
    "yellow": (255, 255, 0),
    "amarillo": (255, 255, 0),
    "orange": (255, 165, 0),
    "naranja": (255, 165, 0),
    "cyan": (0, 255, 255),
    "cian": (0, 255, 255),
    "pink": (255, 105, 180),
    "rosa": (255, 105, 180),
    "purple": (128, 0, 128),
    "morado": (128, 0, 128),
    "violeta": (128, 0, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "gris": (128, 128, 128),
}

NATURAL_COMPOUND_COLOR_TO_RGB = {
    "azul-marino": (18, 42, 84),
    "azul-celeste": (135, 206, 235),
    "verde-oliva": (107, 142, 35),
    "verde-limon": (50, 205, 50),
    "rojo-granate": (128, 0, 32),
    "gris-oscuro": (64, 64, 64),
    "gris-claro": (192, 192, 192),
}

BASE_COLOR_TOKEN_TO_RGB = {
    "blanco": (255, 255, 255),
    "white": (255, 255, 255),
    "negro": (0, 0, 0),
    "black": (0, 0, 0),
    "rojo": (220, 20, 60),
    "red": (220, 20, 60),
    "verde": (0, 170, 0),
    "green": (0, 170, 0),
    "azul": (0, 102, 255),
    "blue": (0, 102, 255),
    "amarillo": (255, 221, 0),
    "yellow": (255, 221, 0),
    "naranja": (255, 140, 0),
    "orange": (255, 140, 0),
    "rosa": (255, 105, 180),
    "pink": (255, 105, 180),
    "morado": (128, 0, 128),
    "violeta": (128, 0, 128),
    "purpura": (128, 0, 128),
    "purple": (128, 0, 128),
    "cian": (0, 180, 200),
    "cyan": (0, 180, 200),
    "turquesa": (64, 224, 208),
    "celeste": (135, 206, 235),
    "marron": (139, 69, 19),
    "brown": (139, 69, 19),
    "beige": (245, 245, 220),
    "gris": (128, 128, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
}

LIGHT_TOKENS = {
    "claro",
    "clara",
    "clarito",
    "clarita",
    "pastel",
    "suave",
    "light",
}
DARK_TOKENS = {"oscuro", "oscura", "oscurito", "oscurita", "dark"}
SAT_UP_TOKENS = {"vivo", "viva", "vibrante", "intenso", "intensa", "neon", "fluor"}
SAT_DOWN_TOKENS = {"apagado", "apagada", "grisaceo", "grisacea", "mate"}
INTENSIFIER_TOKENS = {"muy", "super", "re", "bien", "bastante"}


def parse_args():
    """Parse CLI arguments for the tracking script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full tracking pipeline. Optionally pass a video shortcut from "
            "config.yaml paths.data (e.g. video_prueba_ajustado)."
        )
    )
    parser.add_argument(
        "video_shortcut",
        nargs="?",
        default=None,
        help=(
            "Optional key inside paths.data (config.yaml), e.g. video_prueba_ajustado. "
            "You can also pass a direct video path."
        ),
    )
    parser.add_argument(
        "--team-colors",
        default=None,
        help=(
            "Override de colores por terminal en formato "
            "'{Equipo:color, Otro:color}'. "
            "Acepta lenguaje natural (ej. rojo, verde clarito, azul marino), "
            "HEX (#RRGGBB) o RGB (255,255,255). Si usas nombres de equipo que "
            "no existan en config.yaml, se usarán esos equipos nuevos para este run."
        ),
    )
    parser.add_argument(
        "--team-mode",
        choices=("reference", "auto-bootstrap"),
        default=None,
        help=(
            "Modo de asignación de equipo por color. "
            "'reference' usa equipos de config/--team-colors; "
            "'auto-bootstrap' aprende centroides al inicio y fija equipos por vídeo."
        ),
    )
    parser.add_argument(
        "--team-bootstrap-frames",
        type=int,
        default=None,
        help=(
            "Frames iniciales usados para bootstrap de equipos en modo "
            "'auto-bootstrap'. Si no se indica, usa config.yaml."
        ),
    )
    parser.add_argument(
        "--team-bootstrap-min-samples",
        type=int,
        default=None,
        help=(
            "Muestras mínimas de color para cerrar bootstrap en modo "
            "'auto-bootstrap'. Si no se indica, usa config.yaml."
        ),
    )
    parser.add_argument(
        "--team-bootstrap-min-cluster-samples",
        type=int,
        default=None,
        help=(
            "Muestras mínimas por cluster al cerrar bootstrap en modo "
            "'auto-bootstrap'. Si no se indica, usa config.yaml."
        ),
    )
    return parser.parse_args()


def normalize_token(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized


def parse_rgb_triplet(raw_value):
    numbers = re.findall(r"\d+", raw_value)
    if len(numbers) != 3:
        return None
    values = tuple(int(v) for v in numbers)
    if any(v < 0 or v > 255 for v in values):
        raise ValueError(f"RGB inválido '{raw_value}'. Cada canal debe estar en 0..255.")
    return values


def _clip01(value):
    return min(1.0, max(0.0, float(value)))


def _apply_natural_modifiers(rgb_color, normalized_tokens):
    rgb_01 = tuple(channel / 255.0 for channel in rgb_color)
    h, l, s = colorsys.rgb_to_hls(*rgb_01)

    light_count = sum(token in LIGHT_TOKENS for token in normalized_tokens)
    dark_count = sum(token in DARK_TOKENS for token in normalized_tokens)
    sat_up_count = sum(token in SAT_UP_TOKENS for token in normalized_tokens)
    sat_down_count = sum(token in SAT_DOWN_TOKENS for token in normalized_tokens)
    intensifier_count = sum(token in INTENSIFIER_TOKENS for token in normalized_tokens)

    intensity_factor = 1.0 + (0.5 * intensifier_count)
    light_delta = (0.18 * light_count - 0.18 * dark_count) * intensity_factor
    sat_delta = (0.14 * sat_up_count - 0.14 * sat_down_count) * intensity_factor

    if any(token in {"neon", "fluor"} for token in normalized_tokens):
        l = max(l, 0.58)
        sat_delta += 0.10

    l = _clip01(l + light_delta)
    s = _clip01(s + sat_delta)

    r_out, g_out, b_out = colorsys.hls_to_rgb(h, l, s)
    return (
        int(round(r_out * 255.0)),
        int(round(g_out * 255.0)),
        int(round(b_out * 255.0)),
    )


def parse_natural_color_to_rgb(raw_color):
    normalized_text = normalize_token(raw_color)
    if not normalized_text:
        return None

    if normalized_text in COLOR_NAME_TO_RGB:
        return COLOR_NAME_TO_RGB[normalized_text]
    if normalized_text in NATURAL_COMPOUND_COLOR_TO_RGB:
        return NATURAL_COMPOUND_COLOR_TO_RGB[normalized_text]

    tokens = [token for token in normalized_text.split("-") if token]
    if not tokens:
        return None

    base_colors = []
    for token in tokens:
        if token in BASE_COLOR_TOKEN_TO_RGB:
            base_colors.append(BASE_COLOR_TOKEN_TO_RGB[token])
            continue
        close = difflib.get_close_matches(
            token,
            list(BASE_COLOR_TOKEN_TO_RGB.keys()),
            n=1,
            cutoff=0.8,
        )
        if close:
            base_colors.append(BASE_COLOR_TOKEN_TO_RGB[close[0]])

    if not base_colors:
        return None

    if len(base_colors) == 1:
        base_rgb = base_colors[0]
    else:
        base_rgb = tuple(
            int(round(sum(component[idx] for component in base_colors) / len(base_colors)))
            for idx in range(3)
        )
    return _apply_natural_modifiers(base_rgb, tokens)


def parse_color_to_rgb(raw_color):
    color_text = str(raw_color).strip().strip('"').strip("'")
    normalized_color = normalize_token(color_text)

    if normalized_color in COLOR_NAME_TO_RGB:
        return COLOR_NAME_TO_RGB[normalized_color]

    hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", color_text)
    if hex_match:
        hex_code = hex_match.group(1)
        return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))

    rgb_triplet = parse_rgb_triplet(color_text)
    if rgb_triplet is not None:
        return rgb_triplet

    natural_rgb = parse_natural_color_to_rgb(color_text)
    if natural_rgb is not None:
        return natural_rgb

    available_colors = ", ".join(sorted(COLOR_NAME_TO_RGB.keys()))
    raise ValueError(
        f"Color no soportado '{raw_color}'. "
        "Usa lenguaje natural (ej. 'verde clarito', 'azul marino', "
        "'rojo oscuro'), HEX (#RRGGBB) o RGB. "
        f"Colores base conocidos: {available_colors}."
    )


def rgb_to_lab_opencv(rgb_color):
    rgb_pixel = np.array([[list(rgb_color)]], dtype=np.uint8)
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0, 0]
    return lab_pixel.astype(np.float32)


def parse_team_color_overrides(raw_text):
    text = str(raw_text).strip()
    if not text:
        return []

    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]

    if not text.strip():
        return []

    pairs = []
    for item in text.split(","):
        segment = item.strip()
        if not segment:
            continue
        separator = ":" if ":" in segment else "=" if "=" in segment else None
        if separator is None:
            raise ValueError(
                f"Formato inválido en '{segment}'. "
                "Usa 'Equipo:color' separado por comas."
            )
        team_name, color_name = segment.split(separator, 1)
        team_name = team_name.strip().strip('"').strip("'")
        color_name = color_name.strip().strip('"').strip("'")
        if not team_name or not color_name:
            raise ValueError(
                f"Par inválido '{segment}'. "
                "Equipo y color no pueden estar vacíos."
            )
        pairs.append((team_name, color_name))
    return pairs


def try_resolve_team_name(team_alias, available_team_names):
    normalized_alias = normalize_token(team_alias)
    normalized_team_map = {
        normalize_token(team_name): team_name for team_name in available_team_names
    }

    if normalized_alias in normalized_team_map:
        return normalized_team_map[normalized_alias]

    contains_matches = [
        team_name
        for normalized_name, team_name in normalized_team_map.items()
        if normalized_alias in normalized_name or normalized_name in normalized_alias
    ]
    if len(contains_matches) == 1:
        return contains_matches[0]

    close_matches = difflib.get_close_matches(
        normalized_alias,
        list(normalized_team_map.keys()),
        n=1,
        cutoff=0.65,
    )
    if close_matches:
        return normalized_team_map[close_matches[0]]

    return None


def apply_team_color_overrides(base_team_colors, raw_overrides, logger):
    overrides = parse_team_color_overrides(raw_overrides)
    if not overrides:
        return base_team_colors

    base_team_names = list(base_team_colors.keys())
    resolved_overrides = []
    has_unmatched_team_name = False

    for team_alias, color_value in overrides:
        resolved_team_name = try_resolve_team_name(team_alias, base_team_names)
        if resolved_team_name is None:
            has_unmatched_team_name = True
            resolved_team_name = str(team_alias).strip()
        resolved_overrides.append((team_alias, resolved_team_name, color_value))

    # Si aparece al menos un nombre no reconocido, asumimos que el usuario
    # quiere definir explícitamente los equipos del partido para este run.
    if has_unmatched_team_name:
        updated_team_colors = {}
        override_mode = "explicit_teams"
    else:
        updated_team_colors = dict(base_team_colors)
        override_mode = "partial_update"

    applied = {}
    for team_alias, resolved_team_name, color_value in resolved_overrides:
        rgb_color = parse_color_to_rgb(color_value)
        lab_color = rgb_to_lab_opencv(rgb_color)
        updated_team_colors[resolved_team_name] = lab_color
        applied[resolved_team_name] = {
            "input_team": team_alias,
            "input_color": color_value,
            "rgb": list(rgb_color),
            "lab_opencv": [float(channel) for channel in lab_color],
        }

    logger.info(
        f"Team colors override applied mode={override_mode} "
        f"active_teams={sorted(updated_team_colors.keys())} "
        f"(LAB OpenCV): {json.dumps(applied, ensure_ascii=False)}"
    )
    return updated_team_colors


def resolve_video_path(config, video_shortcut):
    """
    Resolve video path either from a config shortcut (paths.data.<key>)
    or from a direct filesystem path.
    """
    data_paths = config.paths.get("data", {})
    if video_shortcut:
        if video_shortcut in data_paths:
            return str(config.get_path("paths", "data", video_shortcut)), video_shortcut

        candidate = Path(video_shortcut).expanduser()
        if not candidate.is_absolute():
            candidate = (config.project_root / candidate).resolve()
        if candidate.exists():
            return str(candidate), str(candidate)

        available_keys = ", ".join(sorted(data_paths.keys()))
        raise FileNotFoundError(
            f"No se encontró el shortcut/ruta de video '{video_shortcut}'. "
            f"Shortcuts disponibles en paths.data: {available_keys}"
        )

    default_key = (
        "video_prueba_corto"
        if data_paths.get("video_prueba_corto") is not None
        else "video_prueba"
    )
    return str(config.get_path("paths", "data", default_key)), default_key


def build_output_video_path(config, video_path):
    """Build output path using the input video filename."""
    output_dir = config.get_path(
        "paths", "output", "prueba_tracker", create_if_missing=True
    )
    input_stem = Path(video_path).stem or "video"
    output_name = f"{input_stem}_tracking.mp4"
    return str(output_dir / output_name)


def sanitize_video_stem(raw_stem):
    """Sanitize video stem to align with experiments/positions naming convention."""
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def build_tracks_output_paths(config, video_path):
    """
    Build JSON output paths for tracking results:
    - named path: required by experiments/positions notebook
    - legacy path: backwards compatibility
    """
    tracks_dir = config.get_path(
        "paths", "output", "tracks_json", create_if_missing=True
    ) / "tracker"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    video_stem = Path(video_path).stem
    sanitized_stem = sanitize_video_stem(video_stem)
    named_path = tracks_dir / f"{sanitized_stem}_tracks.json"
    legacy_path = tracks_dir / "tracks.json"
    return str(named_path), str(legacy_path)


def build_tracking_metrics_output_paths(config, video_path):
    """
    Build output paths for per-video summary and aggregated metrics dataset.
    """
    tracks_dir = config.get_path(
        "paths", "output", "tracks_json", create_if_missing=True
    ) / "tracker"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = (
        PROJECT_ROOT / "data" / "posiciones_etiquetadas" / "common" / "tracking_metrics.csv"
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    summary_path = tracks_dir / f"{sanitized_stem}_summary.json"
    return str(summary_path), str(dataset_path)


def save_result(tracks, output_path, logger):
    """Saves tracks in JSON format with error handling."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(tracks), f, indent=4, ensure_ascii=False, sort_keys=True)
        logger.info(f"Tracks saved to: {output_path}")
    except IOError as e:
        logger.error(f"Error saving tracks to {output_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving tracks: {e}")


def save_summary(summary, output_path, logger):
    """Save evaluation summary JSON for a single video."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                convert_to_serializable(summary),
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        logger.info(f"Tracking summary saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving tracking summary to {output_path}: {e}")


def _read_csv_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def upsert_tracking_metrics_dataset(
    dataset_path,
    video_path,
    video_source,
    tracks_json_path,
    summary_json_path,
    summary,
    logger,
):
    """
    Upsert a row in tracking_metrics.csv keyed by video_source.
    """
    try:
        dataset_path = Path(dataset_path)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        video_path_obj = Path(video_path)
        row = {
            "video_source": str(video_source),
            "video_filename": video_path_obj.name,
            "video_stem": video_path_obj.stem,
            "video_path": str(video_path),
            "tracks_json_path": str(tracks_json_path),
            "summary_json_path": str(summary_json_path),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in summary.items():
            row[str(key)] = value

        rows = []
        fieldnames = list(row.keys())
        if dataset_path.exists():
            existing_rows, existing_fieldnames = _read_csv_rows(dataset_path)
            rows = [
                existing_row
                for existing_row in existing_rows
                if existing_row.get("video_source") != row["video_source"]
            ]
            for key in existing_fieldnames:
                if key not in fieldnames:
                    fieldnames.append(key)
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        rows.append(row)
        rows.sort(key=lambda item: item.get("video_source", ""))

        with open(dataset_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in rows:
                writer.writerow({name: item.get(name, "") for name in fieldnames})

        logger.info(f"Tracking metrics dataset updated: {dataset_path}")
    except Exception as e:
        logger.error(f"Error updating tracking metrics dataset {dataset_path}: {e}")


if __name__ == "__main__":
    args = parse_args()

    # Load configuration
    config = get_config()
    
    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)
    
    logger.info("Starting football tracking system")
    
    try:
        # Get paths and parameters from config
        MODEL_PATH = str(config.get_path('paths', 'models', 'modelo_base'))
        VIDEO_PATH, video_source = resolve_video_path(config, args.video_shortcut)
        OUTPUT = build_output_video_path(config, VIDEO_PATH)
        OUTPUT_PATH_NAMED, OUTPUT_PATH_LEGACY = build_tracks_output_paths(
            config, VIDEO_PATH
        )
        SUMMARY_PATH, METRICS_DATASET_PATH = build_tracking_metrics_output_paths(
            config, VIDEO_PATH
        )
        ROLE_ARTIFACTS_DIR = build_role_artifacts_output_dir(
            config,
            VIDEO_PATH,
        )
        (
            ROLE_FRAME_CSV_PATH,
            ROLE_PLAYER_CSV_PATH,
            ROLE_GREEDY_CSV_PATH,
        ) = build_role_predictions_output_paths(
            config,
            VIDEO_PATH,
            use_artifacts_dir=True,
        )
        (
            ROLE_FRAME_CSV_PATH_LEGACY,
            ROLE_PLAYER_CSV_PATH_LEGACY,
            ROLE_GREEDY_CSV_PATH_LEGACY,
        ) = build_role_predictions_output_paths(
            config,
            VIDEO_PATH,
            use_artifacts_dir=False,
        )
        
        # Configuration parameters
        CONF = config.get('detection', 'conf_threshold')
        BALL_MIN_CONF = config.get('detection', 'ball_min_conf')
        SHOWKMEANS = config.get('visualization', 'show_kmeans')
        SHOW_OUTPUT = config.get('visualization', 'show_output')
        
        # Tracking configuration
        tracking_cfg = config.tracking
        MAX_TRACKS_PER_CLASS = tracking_cfg.get(
            "max_tracks_per_class",
            {"player": 22, "ball": 1, "referee": 3},
        )
        TRACKER_CONF = {
            "track_thresh": tracking_cfg['track_thresh'],
            "track_buffer": tracking_cfg['track_buffer'],
            "match_thresh": tracking_cfg['match_thresh'],
            "frame_rate": tracking_cfg['frame_rate'],
            "minimum_consecutive_frames": tracking_cfg['minimum_consecutive_frames'],
            "max_total_tracks": tracking_cfg.get("max_total_tracks", 25),
            "enforce_internal_class_limits": tracking_cfg.get(
                "enforce_internal_class_limits", False
            ),
            "team_mismatch_penalty": tracking_cfg.get("team_mismatch_penalty", 1000.0),
            "second_match_threshold": tracking_cfg.get("second_match_threshold", 0.9),
            "unconfirmed_match_threshold": tracking_cfg.get(
                "unconfirmed_match_threshold", 0.8
            ),
            "reassign_motion_factor": tracking_cfg.get("reassign_motion_factor", 1.0),
            "reassign_min_distance": tracking_cfg.get("reassign_min_distance", 25.0),
            "reassign_min_samples": tracking_cfg.get("reassign_min_samples", 3),
            "reassign_motion_growth_cap_frames": tracking_cfg.get(
                "reassign_motion_growth_cap_frames", 12
            ),
            "use_field_position_as_primary_cost": tracking_cfg.get(
                "use_field_position_as_primary_cost", False
            ),
            "use_bbox_center_for_matching": tracking_cfg.get(
                "use_bbox_center_for_matching", True
            ),
            "bbox_center_distance_weight": tracking_cfg.get(
                "bbox_center_distance_weight", 0.7
            ),
            "bbox_center_distance_gate_px": tracking_cfg.get(
                "bbox_center_distance_gate_px", 120.0
            ),
            "lost_time_penalty_weight": tracking_cfg.get(
                "lost_time_penalty_weight", 0.0
            ),
            "lost_time_penalty_max_frames": tracking_cfg.get(
                "lost_time_penalty_max_frames", 10
            ),
            "strict_person_class_separation": tracking_cfg.get(
                "strict_person_class_separation", True
            ),
            "reserve_penalty_spot_seed_players": tracking_cfg.get(
                "reserve_penalty_spot_seed_players", False
            ),
            "reserve_penalty_spot_seed_match_distance_m": tracking_cfg.get(
                "reserve_penalty_spot_seed_match_distance_m", 12.0
            ),
            "require_field_position_for_reassign": tracking_cfg.get(
                "require_field_position_for_reassign", True
            ),
            "max_reassign_lost_frames": tracking_cfg.get(
                "max_reassign_lost_frames", None
            ),
            "max_reassign_lost_frames_by_class": tracking_cfg.get(
                "max_reassign_lost_frames_by_class", {}
            ),
            "motion_std_gate_enabled": tracking_cfg.get(
                "motion_std_gate_enabled", True
            ),
            "motion_std_factor": tracking_cfg.get("motion_std_factor", 10.0),
            "motion_std_min_samples": tracking_cfg.get(
                "motion_std_min_samples", 8
            ),
            "motion_std_floor": tracking_cfg.get("motion_std_floor", 0.5),
            "ball_expected_position_gate_px": tracking_cfg.get(
                "ball_expected_position_gate_px", 90.0
            ),
            "ball_expected_position_gate_growth_per_frame": tracking_cfg.get(
                "ball_expected_position_gate_growth_per_frame", 35.0
            ),
            "ball_expected_position_confidence_relax": tracking_cfg.get(
                "ball_expected_position_confidence_relax", 1.4
            ),
            "ball_size_ratio_per_frame": tracking_cfg.get(
                "ball_size_ratio_per_frame", 1.8
            ),
            "ball_size_min_samples": tracking_cfg.get(
                "ball_size_min_samples", 5
            ),
            "ball_size_std_factor": tracking_cfg.get(
                "ball_size_std_factor", 3.0
            ),
            "ball_size_std_floor": tracking_cfg.get(
                "ball_size_std_floor", 1.0
            ),
            "ball_max_reassign_lost_frames": tracking_cfg.get(
                "ball_max_reassign_lost_frames", 4
            ),
            "ball_high_conf_override": tracking_cfg.get(
                "ball_high_conf_override", 0.6
            ),
            "referee_recovery_max_lost_frames": tracking_cfg.get(
                "referee_recovery_max_lost_frames", 3
            ),
            "referee_recovery_max_distance": tracking_cfg.get(
                "referee_recovery_max_distance", 45.0
            ),
        }
        FIELD_TRACKING_CONF = {
            "enabled": tracking_cfg.get("use_field_positions", True),
            "method": tracking_cfg.get("field_position_method", "pnlcalib"),
            "classes": tracking_cfg.get("field_position_classes", ["player", "goalkeeper"]),
            "field_length_m": tracking_cfg.get("field_length_m", 106.0),
            "field_width_m": tracking_cfg.get("field_width_m", 68.0),
            "max_width": tracking_cfg.get("field_position_max_width", 1280),
            "bottom_offset_ratio": tracking_cfg.get(
                "field_position_bottom_offset_ratio", 0.04
            ),
            "keypoint_threshold": tracking_cfg.get(
                "field_position_keypoint_threshold", 0.3434
            ),
            "line_threshold": tracking_cfg.get(
                "field_position_line_threshold", 0.7867
            ),
            "pnl_refine": tracking_cfg.get("field_position_pnl_refine", True),
            "temporal_blend": tracking_cfg.get(
                "field_position_temporal_blend", 0.20
            ),
            "pixels_per_meter": tracking_cfg.get(
                "field_position_pixels_per_meter", 8
            ),
            "device": tracking_cfg.get("field_position_device"),
            "match_distance_gate_m": tracking_cfg.get(
                "field_position_match_distance_gate_m", 8.0
            ),
            "match_distance_max_lost_frames": tracking_cfg.get(
                "field_position_match_distance_max_lost_frames"
            ),
            "match_distance_cap_m": tracking_cfg.get(
                "field_position_match_distance_cap_m", 6.0
            ),
            "match_distance_growth_mode": tracking_cfg.get(
                "field_position_match_distance_growth_mode", "power"
            ),
            "match_distance_lost_exponent": tracking_cfg.get(
                "field_position_match_distance_lost_exponent", 1.0
            ),
            "match_distance_decay_per_frame": tracking_cfg.get(
                "field_position_match_distance_decay_per_frame", 0.0
            ),
            "match_distance_weight": tracking_cfg.get(
                "field_position_match_distance_weight", 0.25
            ),
            "reassign_min_field_distance_m": tracking_cfg.get(
                "reassign_min_field_distance_m", 4.0
            ),
        }
        color_clustering_cfg = config.color_clustering

        # Team colors
        TEAM_COLORS = config.get_team_colors()
        if args.team_colors:
            TEAM_COLORS = apply_team_color_overrides(
                TEAM_COLORS,
                args.team_colors,
                logger,
            )

        team_mode = args.team_mode or tracking_cfg.get("team_assignment_mode", "reference")
        team_bootstrap_frames = (
            args.team_bootstrap_frames
            if args.team_bootstrap_frames is not None
            else tracking_cfg.get("team_bootstrap_frames", 1)
        )
        team_bootstrap_min_samples = (
            args.team_bootstrap_min_samples
            if args.team_bootstrap_min_samples is not None
            else tracking_cfg.get("team_bootstrap_min_samples", 12)
        )
        team_bootstrap_min_cluster_samples = (
            args.team_bootstrap_min_cluster_samples
            if args.team_bootstrap_min_cluster_samples is not None
            else tracking_cfg.get("team_bootstrap_min_cluster_samples", 4)
        )
        TEAM_DETECTOR_CONF = {
            "confirmation_threshold": color_clustering_cfg.get("confirmation_threshold", 3),
            "color_tolerance": color_clustering_cfg.get("color_tolerance", 25),
            "assignment_mode": team_mode,
            "auto_bootstrap_frames": team_bootstrap_frames,
            "auto_bootstrap_min_samples": team_bootstrap_min_samples,
            "auto_num_teams": tracking_cfg.get("team_bootstrap_num_teams", 2),
            "auto_min_cluster_samples": team_bootstrap_min_cluster_samples,
            "auto_team_name_prefix": tracking_cfg.get("team_auto_name_prefix", "Equipo"),
            "team_candidate_classes": tracking_cfg.get(
                "team_candidate_classes",
                ["player", "goalkeeper"],
            ),
            "shirt_detector_kwargs": {
                "n_clusters": color_clustering_cfg.get("n_clusters", 2),
                "init": color_clustering_cfg.get("init", "k-means++"),
                "n_init": color_clustering_cfg.get("n_init", 10),
                "random_state": color_clustering_cfg.get("random_state", 0),
            },
        }
        
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Video: {VIDEO_PATH}")
        logger.info(f"Video source: {video_source}")
        logger.info(f"Named tracks JSON output: {OUTPUT_PATH_NAMED}")
        logger.info(f"Legacy tracks JSON output: {OUTPUT_PATH_LEGACY}")
        logger.info(f"Summary JSON output: {SUMMARY_PATH}")
        logger.info(f"Role artifacts directory: {ROLE_ARTIFACTS_DIR}")
        logger.info(f"Frame role CSV output: {ROLE_FRAME_CSV_PATH}")
        logger.info(f"Player role CSV output: {ROLE_PLAYER_CSV_PATH}")
        logger.info(f"Greedy role CSV output: {ROLE_GREEDY_CSV_PATH}")
        logger.info(f"Tracking metrics dataset CSV: {METRICS_DATASET_PATH}")
        logger.info(f"Tracking configuration: {TRACKER_CONF}")
        logger.info(f"Field tracking configuration: {FIELD_TRACKING_CONF}")
        logger.info(f"Team detector configuration: {TEAM_DETECTOR_CONF}")
        
        # Run tracking
        tracker = Tracker(
            MODEL_PATH,
            CONF,
            TRACKER_CONF,
            TEAM_COLORS,
            team_detector_conf=TEAM_DETECTOR_CONF,
            ball_min_conf=BALL_MIN_CONF,
            max_tracks_per_class=MAX_TRACKS_PER_CLASS,
            field_tracking_conf=FIELD_TRACKING_CONF,
            project_root=config.project_root,
        )
        online_special_seed_role_assigner = OnlineSpecialSeedRoleAssigner(
            config=config,
            video_path=VIDEO_PATH,
            logger=logger,
        )
        logger.info("Extracting tracks from video...")
        tracks = tracker.get_tracks(
            VIDEO_PATH,
            SHOWKMEANS,
            frame_hook=online_special_seed_role_assigner.on_frame,
        )
        role_postprocess_result = (
            online_special_seed_role_assigner.summary()
            if online_special_seed_role_assigner.enabled
            else None
        )
        if role_postprocess_result is not None:
            logger.info(
                "Online position-role assignment applied: %s player predictions, %s frame predictions",
                role_postprocess_result.get("position_role_player_predictions"),
                role_postprocess_result.get("position_role_frame_predictions"),
            )
        
        # Draw tracks
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors)
        logger.info("Drawing tracks on video...")
        drawer.draw_tracks(tracks, VIDEO_PATH, OUTPUT, show=SHOW_OUTPUT)
        logger.info(f"Video with tracks saved to: {OUTPUT}")
        
        # Evaluation
        logger.info("Evaluating tracks...")
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(["player", "ball"], tracks)
        summary = dict(evaluation["player"]["summary"])
        ball_summary = evaluation.get("ball", {}).get("summary", {})
        summary["ball_coverage"] = float(ball_summary.get("mean_coverage", 0.0))
        summary["ball_tracks"] = int(ball_summary.get("num_tracks", 0))
        if role_postprocess_result is not None:
            summary["position_role_online_applied"] = True
            for key, value in role_postprocess_result.items():
                summary[f"special_seed_{key}"] = value

        # Save tracks JSON
        save_result(tracks, OUTPUT_PATH_NAMED, logger)
        if OUTPUT_PATH_LEGACY != OUTPUT_PATH_NAMED:
            save_result(tracks, OUTPUT_PATH_LEGACY, logger)
        if online_special_seed_role_assigner.enabled:
            role_frame_df, role_player_df, role_greedy_df = (
                online_special_seed_role_assigner.build_role_export_dataframes(tracks)
            )
            save_dataframe_csv(
                role_frame_df,
                ROLE_FRAME_CSV_PATH,
                logger,
                "Frame role predictions CSV",
            )
            copy_output_artifact(
                ROLE_FRAME_CSV_PATH,
                ROLE_FRAME_CSV_PATH_LEGACY,
                logger,
                "Frame role predictions CSV",
            )
            save_dataframe_csv(
                role_player_df,
                ROLE_PLAYER_CSV_PATH,
                logger,
                "Player role summary CSV",
            )
            copy_output_artifact(
                ROLE_PLAYER_CSV_PATH,
                ROLE_PLAYER_CSV_PATH_LEGACY,
                logger,
                "Player role summary CSV",
            )
            save_dataframe_csv(
                role_greedy_df,
                ROLE_GREEDY_CSV_PATH,
                logger,
                "Greedy role diagnostics CSV",
            )
            copy_output_artifact(
                ROLE_GREEDY_CSV_PATH,
                ROLE_GREEDY_CSV_PATH_LEGACY,
                logger,
                "Greedy role diagnostics CSV",
            )
            save_role_visualizations(
                frame_df=role_frame_df,
                player_df=role_player_df,
                config=config,
                video_path=VIDEO_PATH,
                output_dir=ROLE_ARTIFACTS_DIR,
                expected_roles_by_team=online_special_seed_role_assigner.expected_roles_by_team,
                assignment_method=online_special_seed_role_assigner.role_stabilization_expected_roles_assignment,
                min_count=online_special_seed_role_assigner.role_stabilization_expected_roles_min_count,
                min_cumulative_ratio=online_special_seed_role_assigner.role_stabilization_expected_roles_min_ratio,
                min_final_ratio=online_special_seed_role_assigner.role_stabilization_expected_roles_min_final_ratio,
                logger=logger,
                legacy_output_dir=Path(ROLE_FRAME_CSV_PATH_LEGACY).parent,
            )
        save_summary(summary, SUMMARY_PATH, logger)
        upsert_tracking_metrics_dataset(
            METRICS_DATASET_PATH,
            VIDEO_PATH,
            video_source,
            OUTPUT_PATH_NAMED,
            SUMMARY_PATH,
            summary,
            logger,
        )
        
        # Show summary
        logger.info("Evaluation summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        print("\n=== TRACKING SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Process completed successfully")
