import argparse
import colorsys
import csv
import difflib
import json
import math
import re
import shutil
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Permite ejecutar `python scripts/track.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.tracking import Tracker
from football_ai.evaluation import Evaluator
from football_ai.visualization import Drawer
from football_ai.core import get_config, get_logger, Logger, convert_to_serializable

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


def build_role_artifacts_output_dir(config, video_path):
    """Build the per-video directory that stores role CSV/PNG artifacts."""
    role_artifacts_root = config.get_path(
        "paths", "output", "role_artifacts", create_if_missing=True
    )
    role_artifacts_root.mkdir(parents=True, exist_ok=True)

    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    role_artifacts_dir = role_artifacts_root / f"{sanitized_stem}_role_artifacts"
    role_artifacts_dir.mkdir(parents=True, exist_ok=True)
    return role_artifacts_dir


def build_role_predictions_output_paths(config, video_path, use_artifacts_dir=True):
    """Build CSV output paths for online role predictions exported by track.py."""
    sanitized_stem = sanitize_video_stem(Path(video_path).stem)
    if use_artifacts_dir:
        output_dir = build_role_artifacts_output_dir(config, video_path)
    else:
        output_dir = (
            config.get_path("paths", "output", "tracks_json", create_if_missing=True)
            / "tracker"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

    frame_csv_path = output_dir / f"{sanitized_stem}_frame_role_predictions.csv"
    player_csv_path = output_dir / f"{sanitized_stem}_player_role_summary.csv"
    greedy_csv_path = output_dir / f"{sanitized_stem}_greedy_role_diagnostics.csv"
    return str(frame_csv_path), str(player_csv_path), str(greedy_csv_path)


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


def save_dataframe_csv(df, output_path, logger, description):
    """Save a pandas DataFrame to CSV with consistent logging."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"{description} saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving {description} to {output_path}: {e}")


def copy_output_artifact(source_path, target_path, logger, description):
    """Copy an already generated artifact to a secondary path."""
    try:
        source_path = Path(source_path)
        target_path = Path(target_path)
        if source_path.resolve() == target_path.resolve():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        logger.info(f"{description} copied to: {target_path}")
    except Exception as exc:
        logger.error(
            f"Error copying {description} from {source_path} to {target_path}: {exc}"
        )


def _ordered_roles_from_exports(frame_df, player_df):
    seen_roles = []
    for df, col_name in (
        (frame_df, "predicted_role_frame"),
        (player_df, "predicted_role"),
        (player_df, "expected_role_slot"),
    ):
        if df is None or df.empty or col_name not in df.columns:
            continue
        for raw_value in df[col_name].dropna().tolist():
            role = _normalize_role_token(raw_value)
            if role and role not in seen_roles:
                seen_roles.append(role)

    ordered = [role for role in ROLE_PLOT_ORDER if role in seen_roles]
    for role in seen_roles:
        if role not in ordered:
            ordered.append(role)
    return ordered or list(ROLE_PLOT_ORDER)


def _select_stable_role_from_counts(role_counts, confidence_sums):
    best_role = None
    best_priority = None
    for role_name, count in role_counts.items():
        confidence_sum = float(confidence_sums.get(role_name, 0.0))
        mean_confidence = confidence_sum / max(1, int(count))
        priority = (int(count), mean_confidence, str(role_name))
        if best_priority is None or priority > best_priority:
            best_priority = priority
            best_role = str(role_name)
    return best_role


def _ordered_role_labels_from_keys(role_names):
    normalized = []
    for raw_role in role_names:
        role = _normalize_role_token(raw_role)
        if not role or role in normalized or role == "POR":
            continue
        normalized.append(role)

    ordered = [role for role in ROLE_PLOT_ORDER if role in normalized and role != "POR"]
    for role in normalized:
        if role not in ordered and role != "POR":
            ordered.append(role)
    return ordered


def _rank_roles_for_state(state, role_labels):
    observations = max(1, int(state.get("observations", 0)))
    ranked_roles = []
    for normalized_role in role_labels:
        count = int(state.get("role_counts", {}).get(normalized_role, 0))
        if count <= 0:
            continue
        ratio = float(count) / float(observations)
        mean_prob = float(state.get("prob_sums", {}).get(normalized_role, 0.0)) / float(
            observations
        )
        confidence_sum = float(
            state.get("confidence_sums", {}).get(normalized_role, 0.0)
        )
        mean_confidence = confidence_sum / float(count) if count > 0 else 0.0
        ranked_roles.append(
            {
                "role": str(normalized_role),
                "ratio": float(ratio),
                "count": int(count),
                "mean_prob": float(mean_prob),
                "mean_confidence": float(mean_confidence),
                "confidence_sum": float(confidence_sum),
            }
        )

    ranked_roles.sort(
        key=lambda item: (
            item["ratio"],
            item["count"],
            item["mean_prob"],
            item["mean_confidence"],
            item["role"],
        ),
        reverse=True,
    )
    return ranked_roles


def _build_valid_role_ranking_with_invalid_transfer(
    state,
    role_labels,
    valid_roles,
    available_roles=None,
):
    ranked_roles = _rank_roles_for_state(state, role_labels)
    valid_role_set = {_normalize_role_token(role) for role in valid_roles}
    available_counter = (
        Counter(_normalize_role_token(role) for role in available_roles)
        if available_roles is not None
        else None
    )
    transferred_ranking = []
    carry_ratio = 0.0
    carry_count = 0
    carry_prob = 0.0
    carry_confidence_sum = 0.0

    for item in ranked_roles:
        role = str(item["role"])
        if role not in valid_role_set:
            carry_ratio += float(item["ratio"])
            carry_count += int(item["count"])
            carry_prob += float(item["mean_prob"])
            carry_confidence_sum += float(item["confidence_sum"])
            continue

        if available_counter is not None and available_counter.get(role, 0) <= 0:
            carry_ratio += float(item["ratio"])
            carry_count += int(item["count"])
            carry_prob += float(item["mean_prob"])
            carry_confidence_sum += float(item["confidence_sum"])
            continue

        raw_count = int(item["count"])
        raw_confidence_sum = float(item["confidence_sum"])
        effective_count = int(raw_count + carry_count)
        effective_confidence_sum = float(raw_confidence_sum + carry_confidence_sum)
        transferred_ranking.append(
            {
                "role": role,
                "effective_ratio": float(item["ratio"] + carry_ratio),
                "raw_ratio": float(item["ratio"]),
                "effective_count": int(effective_count),
                "raw_count": int(raw_count),
                "effective_mean_prob": float(item["mean_prob"] + carry_prob),
                "raw_mean_prob": float(item["mean_prob"]),
                "effective_mean_confidence": (
                    float(effective_confidence_sum) / float(effective_count)
                    if effective_count > 0
                    else 0.0
                ),
                "raw_mean_confidence": float(item["mean_confidence"]),
                "transferred_ratio": float(carry_ratio),
            }
        )
        carry_ratio = 0.0
        carry_count = 0
        carry_prob = 0.0
        carry_confidence_sum = 0.0

    return transferred_ranking


def _build_available_role_assignment_items(
    state,
    role_labels,
    valid_roles,
    available_roles,
):
    ranking = _build_valid_role_ranking_with_invalid_transfer(
        state=state,
        role_labels=role_labels,
        valid_roles=valid_roles,
        available_roles=available_roles,
    )
    return {str(item["role"]): item for item in ranking}


def _build_direct_role_assignment_item(state, normalized_role, observations):
    raw_count = int(state.get("role_counts", {}).get(normalized_role, 0))
    raw_ratio = float(raw_count) / float(max(1, observations))
    raw_mean_prob = float(state.get("prob_sums", {}).get(normalized_role, 0.0)) / float(
        max(1, observations)
    )
    confidence_sum = float(state.get("confidence_sums", {}).get(normalized_role, 0.0))
    raw_mean_confidence = confidence_sum / float(raw_count) if raw_count > 0 else 0.0
    return {
        "role": str(normalized_role),
        "effective_ratio": float(raw_ratio),
        "raw_ratio": float(raw_ratio),
        "effective_count": int(raw_count),
        "raw_count": int(raw_count),
        "effective_mean_prob": float(raw_mean_prob),
        "raw_mean_prob": float(raw_mean_prob),
        "effective_mean_confidence": float(raw_mean_confidence),
        "raw_mean_confidence": float(raw_mean_confidence),
        "transferred_ratio": 0.0,
    }


def _role_assignment_score_from_item(item, observations):
    return (
        float(item.get("effective_ratio", 0.0))
        + 1e-3 * float(item.get("raw_ratio", 0.0))
        + 1e-6 * float(item.get("effective_mean_prob", 0.0))
        + 1e-9 * float(item.get("raw_mean_prob", 0.0))
        + 1e-12 * float(observations)
    )


def _resolve_remaining_snapshot_assignments(
    pending_candidates,
    available_roles,
    role_labels,
    valid_roles,
    allow_zero_score=False,
    phase="remaining_optimal",
):
    assignments = []
    pending = {int(track_id): state for track_id, state in pending_candidates.items()}
    free_roles = [str(role) for role in available_roles]
    large_cost = 1e6

    while pending and free_roles:
        player_ids = sorted(int(track_id) for track_id in pending.keys())
        role_instances = list(free_roles)
        cost_matrix = np.full(
            (len(player_ids), len(role_instances)),
            large_cost,
            dtype=np.float64,
        )
        pair_payload = {}

        for row_pos, track_id in enumerate(player_ids):
            state = pending[int(track_id)]
            observations = max(1, int(state.get("observations", 0)))
            items_by_role = _build_available_role_assignment_items(
                state=state,
                role_labels=role_labels,
                valid_roles=valid_roles,
                available_roles=role_instances,
            )
            for col_pos, raw_role in enumerate(role_instances):
                normalized_role = _normalize_role_token(raw_role)
                item = items_by_role.get(normalized_role)
                if item is None and allow_zero_score:
                    item = _build_direct_role_assignment_item(
                        state=state,
                        normalized_role=str(normalized_role),
                        observations=int(observations),
                    )
                if item is None:
                    continue
                if (
                    not allow_zero_score
                    and float(item.get("effective_ratio", 0.0)) <= 0.0
                ):
                    continue
                score = _role_assignment_score_from_item(item, observations)
                cost_matrix[row_pos, col_pos] = -float(score)
                pair_payload[(row_pos, col_pos)] = {
                    "track_id": int(track_id),
                    "raw_role": str(raw_role),
                    "normalized_role": str(normalized_role),
                    "state": state,
                    "item": item,
                    "observations": int(observations),
                }

        if not pair_payload:
            break

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_choice = None
        for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist()):
            if cost_matrix[row_pos, col_pos] >= large_cost / 2.0:
                continue
            payload = pair_payload.get((row_pos, col_pos))
            if payload is None:
                continue
            item = payload["item"]
            priority = (
                float(item.get("effective_ratio", 0.0)),
                float(item.get("raw_ratio", 0.0)),
                int(item.get("effective_count", 0)),
                int(item.get("raw_count", 0)),
                float(item.get("effective_mean_prob", 0.0)),
                float(item.get("raw_mean_prob", 0.0)),
                float(item.get("effective_mean_confidence", 0.0)),
                float(item.get("raw_mean_confidence", 0.0)),
                int(payload["observations"]),
                -int(payload["track_id"]),
            )
            if best_choice is None or priority > best_choice[0]:
                best_choice = (
                    priority,
                    payload,
                    list(free_roles),
                )

        if best_choice is None:
            break

        _, payload, available_before = best_choice
        item = payload["item"]
        assignments.append(
            {
                "player_id": int(payload["track_id"]),
                "slot": str(payload["raw_role"]),
                "slot_normalized": str(payload["normalized_role"]),
                "effective_ratio": float(item.get("effective_ratio", 0.0)),
                "final_ratio": float(item.get("raw_ratio", 0.0)),
                "effective_count": int(item.get("effective_count", 0)),
                "count": int(item.get("raw_count", 0)),
                "observations": int(payload["observations"]),
                "available_before": list(available_before),
                "transferred_ratio": float(item.get("transferred_ratio", 0.0)),
                "phase": str(phase),
            }
        )
        pending.pop(int(payload["track_id"]), None)
        try:
            free_roles.remove(str(payload["raw_role"]))
        except ValueError:
            pass

    return assignments, pending, free_roles


def _build_snapshot_role_states(frame_df, cutoff_frame):
    if frame_df is None or frame_df.empty:
        return {}

    states = {}
    prob_cols = [col for col in frame_df.columns if col.startswith("prob_")]
    for row in frame_df.itertuples(index=False):
        try:
            frame_id = int(getattr(row, "frame_id"))
        except (TypeError, ValueError):
            continue
        if frame_id > int(cutoff_frame):
            continue

        team_id = getattr(row, "team_id", None)
        player_id = getattr(row, "player_id", None)
        label = getattr(row, "predicted_role_frame", None)
        if pd.isna(team_id) or pd.isna(player_id) or pd.isna(label):
            continue

        try:
            player_id = int(player_id)
        except (TypeError, ValueError):
            continue

        key = (str(team_id), int(player_id))
        state = states.setdefault(
            key,
            {
                "team_id": str(team_id),
                "player_id": int(player_id),
                "observations": 0,
                "role_counts": {},
                "confidence_sums": {},
                "prob_sums": {},
            },
        )
        role_label = _normalize_role_token(label)
        confidence_value = pd.to_numeric(
            getattr(row, "predicted_role_frame_confidence", 0.0),
            errors="coerce",
        )
        confidence = (
            float(confidence_value)
            if pd.notna(confidence_value) and np.isfinite(float(confidence_value))
            else 0.0
        )
        state["observations"] += 1
        state["role_counts"][role_label] = int(state["role_counts"].get(role_label, 0)) + 1
        state["confidence_sums"][role_label] = float(
            state["confidence_sums"].get(role_label, 0.0)
        ) + confidence
        for prob_col in prob_cols:
            prob_numeric = pd.to_numeric(getattr(row, prob_col, 0.0), errors="coerce")
            prob_value = (
                float(prob_numeric)
                if pd.notna(prob_numeric) and np.isfinite(float(prob_numeric))
                else 0.0
            )
            state["prob_sums"][str(prob_col[5:])] = float(
                state["prob_sums"].get(str(prob_col[5:]), 0.0)
            ) + prob_value
    return states


def _simulate_ratio_priority_snapshot_for_team(
    team_id,
    states,
    expected_roles,
    min_count,
    min_cumulative_ratio,
    min_final_ratio,
):
    team_key = str(team_id)
    available_roles = [
        str(role)
        for role in (expected_roles or [])
        if _normalize_role_token(role) != "POR"
    ]
    ranked_expected_roles = _ordered_role_labels_from_keys(expected_roles or [])
    role_labels = _ordered_role_labels_from_keys(
        set(ranked_expected_roles)
        | {
            role
            for state in states.values()
            if str(state.get("team_id")) == team_key
            for role in list(state.get("role_counts", {}).keys())
            + list(state.get("prob_sums", {}).keys())
        }
    )

    eligible = {}
    low_observations = {}
    for (state_team_id, player_id), state in states.items():
        if str(state_team_id) != team_key:
            continue
        if int(state.get("observations", 0)) >= int(min_count):
            eligible[int(player_id)] = state
        else:
            low_observations[int(player_id)] = state

    pending = dict(eligible)
    steps = []
    while pending and available_roles:
        best_assignment = None
        for player_id, state in pending.items():
            observations = max(1, int(state.get("observations", 0)))
            valid_ranking = _build_valid_role_ranking_with_invalid_transfer(
                state=state,
                role_labels=role_labels,
                valid_roles=ranked_expected_roles,
                available_roles=available_roles,
            )
            chosen = None
            if valid_ranking:
                item = valid_ranking[0]
                normalized_role = str(item["role"])
                chosen = (
                    float(item["effective_ratio"]),
                    float(item["raw_ratio"]),
                    int(item["effective_count"]),
                    int(item["raw_count"]),
                    float(item["effective_mean_prob"]),
                    float(item["raw_mean_prob"]),
                    float(item["effective_mean_confidence"]),
                    float(item["raw_mean_confidence"]),
                    float(item["transferred_ratio"]),
                    str(normalized_role),
                )

            if chosen is None:
                continue

            (
                effective_ratio,
                final_ratio,
                effective_count,
                final_count,
                effective_prob,
                final_prob,
                effective_conf,
                final_conf,
                transferred_ratio,
                normalized_role,
            ) = chosen
            if effective_ratio < float(min_cumulative_ratio) or final_ratio < float(
                min_final_ratio
            ):
                continue

            raw_role = next(
                (
                    str(role)
                    for role in available_roles
                    if _normalize_role_token(role) == normalized_role
                ),
                None,
            )
            if raw_role is None:
                continue

            candidate = (
                float(effective_ratio),
                float(final_ratio),
                int(effective_count),
                int(final_count),
                float(effective_prob),
                float(final_prob),
                float(effective_conf),
                float(final_conf),
                int(observations),
                -int(player_id),
            )
            if best_assignment is None or candidate > best_assignment[0]:
                best_assignment = (
                    candidate,
                    int(player_id),
                    str(raw_role),
                    str(normalized_role),
                    state,
                    list(available_roles),
                    float(transferred_ratio),
                )

        if best_assignment is None:
            break

        (
            candidate,
            player_id,
            raw_role,
            normalized_role,
            state,
            available_before,
            transferred_ratio,
        ) = best_assignment
        observations = max(1, int(state.get("observations", 0)))
        steps.append(
            {
                "team_id": team_key,
                "step_idx": len(steps),
                "player_id": int(player_id),
                "slot": str(raw_role),
                "slot_normalized": str(normalized_role),
                "effective_ratio": float(candidate[0]),
                "final_ratio": float(candidate[1]),
                "effective_count": int(candidate[2]),
                "count": int(candidate[3]),
                "observations": int(observations),
                "available_before": list(available_before),
                "transferred_ratio": float(transferred_ratio),
                "phase": "threshold",
            }
        )
        pending.pop(int(player_id), None)
        try:
            available_roles.remove(str(raw_role))
        except ValueError:
            pass

    residual_assignments, pending, available_roles = _resolve_remaining_snapshot_assignments(
        pending_candidates=pending,
        available_roles=available_roles,
        role_labels=role_labels,
        valid_roles=ranked_expected_roles,
    )
    for assignment in residual_assignments:
        steps.append(
            {
                "team_id": team_key,
                "step_idx": len(steps),
                "player_id": int(assignment["player_id"]),
                "slot": str(assignment["slot"]),
                "slot_normalized": str(assignment["slot_normalized"]),
                "effective_ratio": float(assignment["effective_ratio"]),
                "final_ratio": float(assignment["final_ratio"]),
                "effective_count": int(assignment["effective_count"]),
                "count": int(assignment["count"]),
                "observations": int(assignment["observations"]),
                "available_before": list(assignment["available_before"]),
                "transferred_ratio": float(assignment["transferred_ratio"]),
                "phase": str(assignment.get("phase", "remaining_optimal")),
            }
        )

    final_fill_candidates = dict(pending)
    final_fill_candidates.update(low_observations)
    final_fill_assignments, final_fill_candidates, available_roles = (
        _resolve_remaining_snapshot_assignments(
            pending_candidates=final_fill_candidates,
            available_roles=available_roles,
            role_labels=role_labels,
            valid_roles=ranked_expected_roles,
            allow_zero_score=True,
            phase="fill_remaining",
        )
    )
    for assignment in final_fill_assignments:
        steps.append(
            {
                "team_id": team_key,
                "step_idx": len(steps),
                "player_id": int(assignment["player_id"]),
                "slot": str(assignment["slot"]),
                "slot_normalized": str(assignment["slot_normalized"]),
                "effective_ratio": float(assignment["effective_ratio"]),
                "final_ratio": float(assignment["final_ratio"]),
                "effective_count": int(assignment["effective_count"]),
                "count": int(assignment["count"]),
                "observations": int(assignment["observations"]),
                "available_before": list(assignment["available_before"]),
                "transferred_ratio": float(assignment["transferred_ratio"]),
                "phase": str(assignment.get("phase", "fill_remaining")),
            }
        )
    pending = {
        int(player_id): state
        for player_id, state in pending.items()
        if int(player_id) in final_fill_candidates
    }
    low_observations = {
        int(player_id): state
        for player_id, state in low_observations.items()
        if int(player_id) in final_fill_candidates
    }

    unresolved = []
    remaining_counter = Counter(_normalize_role_token(role) for role in available_roles)
    for category, candidates in (
        ("threshold", pending),
        ("low_observations", low_observations),
    ):
        for player_id, state in sorted(candidates.items()):
            observations = max(1, int(state.get("observations", 0)))
            valid_ranking = _build_valid_role_ranking_with_invalid_transfer(
                state=state,
                role_labels=role_labels,
                valid_roles=ranked_expected_roles,
                available_roles=available_roles,
            )
            best_available_role = None
            best_available_count = 0
            best_available_ratio = 0.0
            best_available_effective_ratio = 0.0
            transferred_ratio = 0.0
            if valid_ranking:
                item = valid_ranking[0]
                normalized_role = str(item["role"])
                if remaining_counter.get(normalized_role, 0) > 0:
                    best_available_role = str(normalized_role)
                    best_available_count = int(item["raw_count"])
                    best_available_ratio = float(item["raw_ratio"])
                    best_available_effective_ratio = float(item["effective_ratio"])
                    transferred_ratio = float(item["transferred_ratio"])

            dominant_role = _select_stable_role_from_counts(
                state.get("role_counts", {}),
                state.get("confidence_sums", {}),
            )
            dominant_count = int(state.get("role_counts", {}).get(dominant_role, 0))
            unresolved.append(
                {
                    "player_id": int(player_id),
                    "category": str(category),
                    "observations": int(observations),
                    "dominant_role": dominant_role,
                    "dominant_ratio": float(dominant_count) / float(observations),
                    "best_available_role": best_available_role,
                    "best_available_count": int(best_available_count),
                    "best_available_ratio": float(best_available_ratio),
                    "best_available_effective_ratio": float(best_available_effective_ratio)
                    if best_available_role is not None
                    else 0.0,
                    "best_available_transferred_ratio": float(transferred_ratio),
                }
            )

    return steps, unresolved, list(available_roles)


def save_role_assignment_overview_plot(
    frame_df,
    player_df,
    output_path,
    logger,
    cutoff_frame=400,
):
    if frame_df is None or frame_df.empty or player_df is None or player_df.empty:
        return

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_roles = _ordered_roles_from_exports(frame_df, player_df)
        cutoff_df = frame_df[
            pd.to_numeric(frame_df.get("frame_id"), errors="coerce").fillna(-1)
            <= int(cutoff_frame)
        ].copy()
        dist = (
            cutoff_df.groupby(["team_id", "player_id", "predicted_role_frame"])
            .size()
            .rename("count")
            .reset_index()
        )
        if dist.empty:
            return
        observations = (
            dist.groupby(["team_id", "player_id"])["count"]
            .sum()
            .rename("observations")
            .reset_index()
        )
        dist = dist.merge(observations, on=["team_id", "player_id"], how="left")
        dist["ratio"] = dist["count"] / dist["observations"]
        pivot = (
            dist.pivot_table(
                index=["team_id", "player_id"],
                columns="predicted_role_frame",
                values="ratio",
                fill_value=0.0,
            )
            .reset_index()
        )
        for role in ordered_roles:
            if role not in pivot.columns:
                pivot[role] = 0.0

        plot_df = player_df.copy()
        plot_df = plot_df.merge(pivot, on=["team_id", "player_id"], how="left")
        for role in ordered_roles:
            plot_df[role] = plot_df.get(role, 0.0).fillna(0.0)

        for idx, row in plot_df.iterrows():
            if float(sum(float(row.get(role, 0.0)) for role in ordered_roles)) == 0.0:
                predicted_role = _normalize_role_token(row.get("predicted_role"))
                if predicted_role == "POR" and "POR" in ordered_roles:
                    plot_df.at[idx, "POR"] = 1.0

        teams = [team for team in plot_df["team_id"].dropna().astype(str).unique().tolist()]
        teams.sort()
        if not teams:
            return

        fig, axes = plt.subplots(
            1,
            len(teams),
            figsize=(22, max(8, 0.7 * max(plot_df.groupby("team_id").size().max(), 8))),
            sharex=True,
        )
        if len(teams) == 1:
            axes = [axes]

        for ax, team_id in zip(axes, teams):
            team_df = plot_df[plot_df["team_id"].astype(str) == str(team_id)].copy()
            team_df = team_df.sort_values("player_id").reset_index(drop=True)
            y_positions = np.arange(len(team_df))
            left = np.zeros(len(team_df))

            for role in ordered_roles:
                vals = team_df[role].to_numpy(dtype=float)
                if np.any(vals > 0):
                    ax.barh(
                        y_positions,
                        vals,
                        left=left,
                        color=ROLE_PLOT_COLORS.get(role, "#adb5bd"),
                        edgecolor="white",
                        height=0.72,
                    )
                    left += vals

            for row_idx, row in team_df.iterrows():
                final_role = row.get("expected_role_slot")
                if pd.isna(final_role) or not str(final_role).strip():
                    final_role = row.get("predicted_role")
                final_role = _normalize_role_token(final_role)
                if not final_role:
                    continue
                cumulative = 0.0
                center_x = 0.98
                for role in ordered_roles:
                    width = float(row.get(role, 0.0))
                    if role == final_role:
                        center_x = cumulative + (width / 2.0 if width > 0 else 0.01)
                        break
                    cumulative += width

                has_expected_slot = pd.notna(row.get("expected_role_slot")) and str(
                    row.get("expected_role_slot")
                ).strip()
                marker = "D" if has_expected_slot else "X"
                face = (
                    ROLE_PLOT_COLORS.get(final_role, "#000000")
                    if has_expected_slot
                    else "white"
                )
                ax.scatter(
                    center_x,
                    row_idx,
                    marker=marker,
                    s=90 if has_expected_slot else 110,
                    c=face,
                    edgecolors="#212529",
                    linewidths=1.2,
                    zorder=5,
                )

                top_roles = sorted(
                    [
                        (role, float(row.get(role, 0.0)))
                        for role in ordered_roles
                        if float(row.get(role, 0.0)) > 0
                    ],
                    key=lambda item: (-item[1], item[0]),
                )[:3]
                summary_text = " | ".join(
                    f"{role} {ratio:.0%}" for role, ratio in top_roles
                )
                ax.text(1.01, row_idx, summary_text, va="center", ha="left", fontsize=9)

            ax.set_yticks(y_positions)
            ax.set_yticklabels([str(int(player_id)) for player_id in team_df["player_id"]])
            ax.invert_yaxis()
            ax.set_xlim(0, 1.18)
            ax.set_xlabel(f"Proporción detectada hasta frame {int(cutoff_frame)}")
            ax.set_title(str(team_id))
            ax.grid(axis="x", linestyle=":", alpha=0.35)

        axes[0].set_ylabel("Jugador")
        fig.suptitle(
            f"Roles detectados hasta frame {int(cutoff_frame)} vs posición final",
            fontsize=16,
            y=0.995,
        )
        legend_roles = [
            Line2D([0], [0], color=ROLE_PLOT_COLORS.get(role, "#adb5bd"), lw=8, label=role)
            for role in ordered_roles
            if any(plot_df[role] > 0)
        ]
        legend_markers = [
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                label="Plaza esperada asignada",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label="Sin plaza esperada",
                markerfacecolor="white",
                markeredgecolor="#212529",
                markersize=8,
            ),
        ]
        fig.legend(
            handles=legend_roles + legend_markers,
            loc="lower center",
            ncol=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Role assignment overview plot saved to: {output_path}")
    except Exception as exc:
        logger.error(f"Error generating role assignment overview plot: {exc}")


def save_ratio_priority_step_by_step_plots(
    frame_df,
    output_dir,
    video_stem,
    expected_roles_by_team,
    min_count,
    min_cumulative_ratio,
    min_final_ratio,
    logger,
    cutoff_frame=400,
):
    if frame_df is None or frame_df.empty or not expected_roles_by_team:
        return

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        states = _build_snapshot_role_states(frame_df, cutoff_frame=cutoff_frame)
        if not states:
            return

        for team_id, expected_roles in expected_roles_by_team.items():
            steps, unresolved, remaining_roles = _simulate_ratio_priority_snapshot_for_team(
                team_id=team_id,
                states=states,
                expected_roles=expected_roles,
                min_count=min_count,
                min_cumulative_ratio=min_cumulative_ratio,
                min_final_ratio=min_final_ratio,
            )
            team_slug = sanitize_video_stem(str(team_id)).lower()
            output_path = (
                output_dir / f"{video_stem}_{team_slug}_ratio_priority_step_by_step.png"
            )

            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2])
            ax_top = fig.add_subplot(gs[0])
            ax_bottom = fig.add_subplot(gs[1])
            ax_top.axis("off")
            ax_top.set_title(
                f"{team_id}: ratio_priority snapshot paso a paso",
                fontsize=16,
                pad=12,
            )

            if steps:
                y = 0.5
                for idx, step in enumerate(steps):
                    phase_value = str(step.get("phase"))
                    if phase_value == "threshold":
                        phase_label = "umbral"
                    elif phase_value == "fill_remaining":
                        phase_label = "relleno final"
                    else:
                        phase_label = "relleno óptimo"
                    text = (
                        f"Paso {idx + 1}\n"
                        f"{phase_label}\n"
                        f"jugador {step['player_id']} -> {step['slot']}\n"
                        f"ratio acum. {step['effective_ratio']:.3f}\n"
                        f"ratio final {step['final_ratio']:.3f}"
                        f" ({step['count']}/{step['observations']})\n"
                        f"libres antes: {', '.join(step['available_before'])}"
                    )
                    ax_top.text(
                        idx,
                        y,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.45",
                            fc="#f8f9fa",
                            ec="#495057",
                            lw=1.2,
                        ),
                    )
                    if idx < len(steps) - 1:
                        ax_top.add_patch(
                            FancyArrowPatch(
                                (idx + 0.38, y),
                                (idx + 0.62, y),
                                arrowstyle="->",
                                mutation_scale=14,
                                lw=1.5,
                                color="#6c757d",
                            )
                        )
                ax_top.set_xlim(-0.6, len(steps) - 0.4)
                ax_top.set_ylim(0, 1)
            else:
                ax_top.text(
                    0.5,
                    0.5,
                    "No hubo asignaciones por ratio_priority en la foto global.",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            ax_bottom.set_title(
                "Jugadores sin plaza táctica y mejor camino restante",
                fontsize=13,
                pad=10,
            )
            if unresolved:
                players = [str(item["player_id"]) for item in unresolved]
                cumulative_vals = [
                    float(item["best_available_effective_ratio"]) for item in unresolved
                ]
                final_vals = [float(item["best_available_ratio"]) for item in unresolved]
                colors = [
                    "#adb5bd" if item["category"] == "low_observations" else "#e9c46a"
                    for item in unresolved
                ]
                bars = ax_bottom.bar(
                    players,
                    cumulative_vals,
                    color=colors,
                    alpha=0.85,
                    label="Ratio acumulado hasta plaza libre",
                )
                ax_bottom.scatter(
                    players,
                    final_vals,
                    color="#d62828",
                    zorder=5,
                    label="Ratio de la plaza final libre",
                )
                ax_bottom.axhline(
                    float(min_cumulative_ratio),
                    color="#1d3557",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Umbral acumulado {float(min_cumulative_ratio):.2f}",
                )
                ax_bottom.axhline(
                    float(min_final_ratio),
                    color="#e76f51",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Umbral final {float(min_final_ratio):.2f}",
                )
                ax_bottom.set_ylim(0, 1.05)
                ax_bottom.set_ylabel("Ratio")
                ax_bottom.grid(axis="y", linestyle=":", alpha=0.35)
                for bar, item in zip(bars, unresolved):
                    available_role = item["best_available_role"] or "sin plaza"
                    extra = (
                        "menos de 200 obs"
                        if item["category"] == "low_observations"
                        else f"dominante {item['dominant_role']} {item['dominant_ratio']:.3f}"
                    )
                    ax_bottom.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.03,
                        f"{extra}\n"
                        f"mejor libre {available_role} "
                        f"{item['best_available_ratio']:.3f}"
                        f" (+{item['best_available_transferred_ratio']:.3f})",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
                ax_bottom.legend(loc="upper right", frameon=False)
            else:
                ax_bottom.axis("off")
                ax_bottom.text(
                    0.5,
                    0.5,
                    "Sin jugadores pendientes tras la asignación snapshot.",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            fig.text(
                0.01,
                0.01,
                "Plazas finales sin cubrir: " + (", ".join(remaining_roles) or "ninguna"),
                fontsize=10,
            )
            fig.tight_layout(rect=[0, 0.03, 1, 1])
            fig.savefig(output_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Ratio-priority step-by-step plot saved to: {output_path}")
    except Exception as exc:
        logger.error(f"Error generating ratio-priority step-by-step plots: {exc}")


def save_role_visualizations(
    frame_df,
    player_df,
    config,
    video_path,
    output_dir,
    expected_roles_by_team,
    assignment_method,
    min_count,
    min_cumulative_ratio,
    min_final_ratio,
    logger,
    legacy_output_dir=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_output_dir = Path(legacy_output_dir) if legacy_output_dir is not None else None
    video_stem = sanitize_video_stem(Path(video_path).stem)
    cutoff_frame = int(config.tracking.get("role_stabilization_window_frames", 400))

    overview_name = f"{video_stem}_role_assignment_vs_detected_pre{int(cutoff_frame)}.png"
    overview_path = output_dir / overview_name
    save_role_assignment_overview_plot(
        frame_df=frame_df,
        player_df=player_df,
        output_path=overview_path,
        logger=logger,
        cutoff_frame=cutoff_frame,
    )
    if legacy_output_dir is not None:
        copy_output_artifact(
            overview_path,
            legacy_output_dir / overview_name,
            logger,
            "Role assignment overview plot",
        )

    if str(assignment_method).strip().lower() == "ratio_priority":
        save_ratio_priority_step_by_step_plots(
            frame_df=frame_df,
            output_dir=output_dir,
            video_stem=video_stem,
            expected_roles_by_team=expected_roles_by_team,
            min_count=min_count,
            min_cumulative_ratio=min_cumulative_ratio,
            min_final_ratio=min_final_ratio,
            logger=logger,
            cutoff_frame=cutoff_frame,
        )
        if legacy_output_dir is not None:
            for team_id in (expected_roles_by_team or {}).keys():
                team_label = sanitize_video_stem(str(team_id)).lower()
                step_plot_name = (
                    f"{video_stem}_{team_label}_ratio_priority_step_by_step.png"
                )
                copy_output_artifact(
                    output_dir / step_plot_name,
                    legacy_output_dir / step_plot_name,
                    logger,
                    "Ratio-priority step-by-step plot",
                )


DEFAULT_SPECIAL_SEED_CANONICAL_IDS = (1, 2)
DEFAULT_SPECIAL_SEED_DEFENDER_ROLES = (
    "CD",
    "CI",
    "LD",
    "LI",
    "DFC_DER",
    "DFC_IZQ",
    "DFC_CENT",
)
DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH = (
    "models/positions/set_transformer/20260317_211507/set_transformer_checkpoint.pt"
)
DEFAULT_EXPECTED_ROLES_BY_TEAM = {
    "Real Madrid": [
        "POR",
        "LD",
        "LI",
        "DFC_DER",
        "DFC_IZQ",
        "DFC_CENT",
        "MC",
        "MI",
        "MD",
        "DC",
        "DC",
    ],
    "Wolfsburgo": [
        "POR",
        "LD",
        "LI",
        "DFC_DER",
        "DFC_IZQ",
        "DFC_CENT",
        "MC",
        "MI",
        "MD",
        "DC",
        "DC",
    ],
}

ROLE_PLOT_ORDER = (
    "POR",
    "LI",
    "CI",
    "DFC_IZQ",
    "DFC_CENT",
    "DFC_DER",
    "LD",
    "CD",
    "MC",
    "MC_IZQ",
    "MC_DCHO",
    "MI",
    "MD",
    "EI",
    "ED",
    "DC",
    "DC_IZQ",
    "DC_DCHO",
)

ROLE_PLOT_COLORS = {
    "POR": "#6c757d",
    "LI": "#2a9d8f",
    "CI": "#2f9e44",
    "DFC_IZQ": "#3a86ff",
    "DFC_CENT": "#4361ee",
    "DFC_DER": "#4895ef",
    "LD": "#4cc9f0",
    "CD": "#72efdd",
    "MC": "#f4a261",
    "MC_IZQ": "#f4a261",
    "MC_DCHO": "#f4a261",
    "MI": "#e9c46a",
    "MD": "#f6bd60",
    "EI": "#e76f51",
    "ED": "#f28482",
    "DC": "#d62828",
    "DC_IZQ": "#d62828",
    "DC_DCHO": "#d62828",
}


def _normalize_role_token(value):
    token = str(value).strip().upper()
    token = token.replace("-", "_").replace(" ", "_")
    token = "_".join(part for part in token.split("_") if part)
    return token


def _role_base_token(value):
    token = _normalize_role_token(value)
    if token in {"MC_IZQ", "MC_DCHO"}:
        return "MC"
    if token in {"DC_IZQ", "DC_DCHO"}:
        return "DC"
    return token


def _format_role_overlay_label(value):
    token = _normalize_role_token(value)
    if token in {"MC_IZQ", "MC_DCHO", "DC_IZQ", "DC_DCHO"}:
        return token
    return str(value)


def _normalize_expected_roles_mapping(raw_mapping):
    if not isinstance(raw_mapping, dict):
        return None

    normalized = {}
    for team_id, raw_roles in raw_mapping.items():
        if raw_roles is None:
            continue
        if not isinstance(raw_roles, (list, tuple)):
            raise ValueError(
                f"tracking.expected_roles_by_team[{team_id!r}] debe ser lista/tupla, no {type(raw_roles).__name__}."
            )
        roles = [str(role).strip() for role in raw_roles if str(role).strip()]
        if roles:
            normalized[str(team_id)] = roles
    return normalized or None


def _resolve_expected_roles_by_team_from_config(config):
    tracking_cfg = config.tracking
    configured = _normalize_expected_roles_mapping(
        tracking_cfg.get("expected_roles_by_team")
    )
    if configured:
        return configured

    configured_team_names = {str(team_name) for team_name in config.get_team_names()}
    default_team_names = set(DEFAULT_EXPECTED_ROLES_BY_TEAM.keys())
    if default_team_names.issubset(configured_team_names):
        return {
            str(team_id): [str(role) for role in roles]
            for team_id, roles in DEFAULT_EXPECTED_ROLES_BY_TEAM.items()
        }
    return None


def _safe_field_position_m(track_data):
    field_position = track_data.get("field_position_m")
    if field_position is None:
        return None
    arr = np.asarray(field_position, dtype=np.float32).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return float(arr[0]), float(arr[1])


def _resolve_project_relative_path(project_root, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _special_seed_frame_track_keys(frame_tracks, special_id):
    keys = []
    if special_id in frame_tracks:
        keys.append(special_id)
    special_id_str = str(int(special_id))
    if special_id_str in frame_tracks:
        keys.append(special_id_str)
    return keys


def _assign_special_seed_teams_from_nearest_defender(
    tracks,
    special_ids,
    defender_roles,
    logger,
):
    special_ids = tuple(int(track_id) for track_id in special_ids)
    defender_roles = {_normalize_role_token(role) for role in defender_roles}
    last_team_by_special_id = {}
    assigned_frames = 0
    carry_frames = 0
    unassigned_frames = 0

    max_frames = max(
        len(tracks.get("player", [])),
        len(tracks.get("goalkeeper", [])),
    )
    person_classes = ("player", "goalkeeper")

    for frame_id in range(max_frames):
        defender_candidates = []
        for class_name in person_classes:
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                continue
            frame_tracks = class_frames[frame_id]
            if not isinstance(frame_tracks, dict):
                continue
            for track_id_raw, track_data in frame_tracks.items():
                try:
                    track_id = int(track_id_raw)
                except (TypeError, ValueError):
                    continue
                if track_id in special_ids:
                    continue
                team_id = track_data.get("team")
                if team_id is None:
                    continue
                field_position = _safe_field_position_m(track_data)
                if field_position is None:
                    continue
                predicted_role = (
                    track_data.get("predicted_role")
                    or track_data.get("predicted_role_frame")
                )
                if _normalize_role_token(predicted_role) not in defender_roles:
                    continue
                defender_candidates.append(
                    {
                        "track_id": track_id,
                        "team": str(team_id),
                        "position": field_position,
                    }
                )

        for class_name in person_classes:
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                continue
            frame_tracks = class_frames[frame_id]
            if not isinstance(frame_tracks, dict):
                continue

            for special_id in special_ids:
                for track_key in _special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if not isinstance(track_data, dict):
                        continue
                    field_position = _safe_field_position_m(track_data)
                    if field_position is None:
                        continue

                    best_candidate = None
                    best_distance_m = None
                    for defender in defender_candidates:
                        distance_m = math.dist(field_position, defender["position"])
                        if best_distance_m is None or distance_m < best_distance_m:
                            best_distance_m = distance_m
                            best_candidate = defender

                    if best_candidate is not None:
                        resolved_team = str(best_candidate["team"])
                        track_data["team"] = resolved_team
                        track_data["special_team_assignment_source"] = "nearest_defender_role"
                        track_data["special_team_assignment_distance_m"] = float(best_distance_m)
                        track_data["special_team_assignment_track_id"] = int(best_candidate["track_id"])
                        last_team_by_special_id[int(special_id)] = resolved_team
                        assigned_frames += 1
                    elif int(special_id) in last_team_by_special_id:
                        track_data["team"] = last_team_by_special_id[int(special_id)]
                        track_data["special_team_assignment_source"] = "nearest_defender_role_carry"
                        track_data["special_team_assignment_distance_m"] = None
                        track_data["special_team_assignment_track_id"] = None
                        carry_frames += 1
                    else:
                        track_data["special_team_assignment_source"] = "unassigned"
                        track_data["special_team_assignment_distance_m"] = None
                        track_data["special_team_assignment_track_id"] = None
                        unassigned_frames += 1

    logger.info(
        "Special seed team assignment: assigned=%s carry=%s unassigned=%s ids=%s",
        assigned_frames,
        carry_frames,
        unassigned_frames,
        list(special_ids),
    )
    return {
        "assigned_frames": int(assigned_frames),
        "carry_frames": int(carry_frames),
        "unassigned_frames": int(unassigned_frames),
        "special_ids": [int(track_id) for track_id in special_ids],
    }


def apply_special_seed_role_team_assignment(
    tracks,
    video_path,
    config,
    logger,
):
    tracking_cfg = config.tracking
    if not tracking_cfg.get("special_seed_role_team_assignment_enabled", True):
        return tracks, None

    model_path = _resolve_project_relative_path(
        config.project_root,
        tracking_cfg.get(
            "special_seed_role_model_path",
            DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH,
        ),
    )
    if not model_path.exists():
        logger.warning(
            "Se omite special_seed_role_team_assignment: no existe el checkpoint %s",
            model_path,
        )
        return tracks, None

    try:
        from experiments.positions.set_transformer_pipeline import (
            predict_roles_for_tracks_payload,
        )
    except Exception as exc:
        logger.warning(
            "No se pudo importar el pipeline de roles posicionales: %s",
            exc,
        )
        return tracks, None

    logger.info(
        "Running position-role postprocess for special seeds with checkpoint: %s",
        model_path,
    )
    try:
        role_result = predict_roles_for_tracks_payload(
            model_path=model_path,
            tracks=tracks,
            video_path=Path(video_path),
            project_root=config.project_root,
        )
    except Exception as exc:
        logger.warning(
            "Falló el postproceso de roles posicionales para special seeds: %s",
            exc,
        )
        return tracks, None
    enriched_tracks = role_result["tracks_with_roles"]
    assignment_summary = _assign_special_seed_teams_from_nearest_defender(
        tracks=enriched_tracks,
        special_ids=tracking_cfg.get(
            "special_seed_canonical_ids",
            list(DEFAULT_SPECIAL_SEED_CANONICAL_IDS),
        ),
        defender_roles=tracking_cfg.get(
            "special_seed_defender_roles",
            list(DEFAULT_SPECIAL_SEED_DEFENDER_ROLES),
        ),
        logger=logger,
    )
    role_result["special_seed_team_assignment"] = assignment_summary
    return enriched_tracks, role_result


class OnlineSpecialSeedRoleAssigner:
    def __init__(self, config, video_path, logger):
        self.config = config
        self.video_path = Path(video_path)
        self.logger = logger
        tracking_cfg = config.tracking
        self.enabled = bool(
            tracking_cfg.get("special_seed_role_team_assignment_enabled", True)
        )
        self.special_ids = tuple(
            int(track_id)
            for track_id in tracking_cfg.get(
                "special_seed_canonical_ids",
                list(DEFAULT_SPECIAL_SEED_CANONICAL_IDS),
            )
        )
        self.role_stabilization_window_frames = max(
            1,
            int(tracking_cfg.get("role_stabilization_window_frames", 400)),
        )
        self.role_stabilization_min_observations = max(
            1,
            int(tracking_cfg.get("role_stabilization_min_observations", 400)),
        )
        self.role_stabilization_vote_ratio = float(
            tracking_cfg.get("role_stabilization_vote_ratio", 0.7)
        )
        self.role_stabilization_expected_roles_assignment = str(
            tracking_cfg.get(
                "role_stabilization_expected_roles_assignment",
                "hungarian",
            )
        ).strip().lower()
        if self.role_stabilization_expected_roles_assignment not in {
            "hungarian",
            "greedy",
            "ratio_priority",
        }:
            self.logger.warning(
                "Valor no soportado para tracking.role_stabilization_expected_roles_assignment=%r; se usara 'hungarian'.",
                self.role_stabilization_expected_roles_assignment,
            )
            self.role_stabilization_expected_roles_assignment = "hungarian"
        self.role_stabilization_expected_roles_min_ratio = float(
            tracking_cfg.get(
                "role_stabilization_expected_roles_min_ratio",
                0.40,
            )
        )
        self.role_stabilization_expected_roles_min_final_ratio = float(
            tracking_cfg.get(
                "role_stabilization_expected_roles_min_final_ratio",
                self.role_stabilization_expected_roles_min_ratio,
            )
        )
        self.role_stabilization_expected_roles_min_count = max(
            1,
            int(
                tracking_cfg.get(
                    "role_stabilization_expected_roles_min_count",
                    200,
                )
            ),
        )
        self.defender_roles = {
            _normalize_role_token(role)
            for role in tracking_cfg.get(
                "special_seed_defender_roles",
                list(DEFAULT_SPECIAL_SEED_DEFENDER_ROLES),
            )
        }
        self.expected_roles_by_team = _resolve_expected_roles_by_team_from_config(config)
        self.last_team_by_special_id = {}
        self.prev_positions = {}
        self.role_state_by_track_id = {}
        self.role_stabilization_snapshot_done = False
        self.raw_frame_prediction_rows = []
        self.greedy_diagnostic_rows = []
        self.stats = {
            "processed_frames": 0,
            "frames_with_role_predictions": 0,
            "position_role_frame_predictions": 0,
            "position_role_player_predictions": 0,
            "special_seed_assigned_frames": 0,
            "special_seed_carry_frames": 0,
            "special_seed_unassigned_frames": 0,
            "stable_role_tracks_frozen": 0,
            "special_ids": [int(track_id) for track_id in self.special_ids],
        }
        self.role_session = None
        if not self.enabled:
            return

        model_path = _resolve_project_relative_path(
            config.project_root,
            tracking_cfg.get(
                "special_seed_role_model_path",
                DEFAULT_SPECIAL_SEED_ROLE_MODEL_PATH,
            ),
        )
        if not model_path.exists():
            self.logger.warning(
                "Se desactiva online special-seed role assignment: no existe el checkpoint %s",
                model_path,
            )
            self.enabled = False
            return

        try:
            from experiments.positions.set_transformer_pipeline import (
                OnlineRoleInferenceSession,
            )
        except Exception as exc:
            self.logger.warning(
                "Se desactiva online special-seed role assignment: no se pudo importar OnlineRoleInferenceSession (%s)",
                exc,
            )
            self.enabled = False
            return

        try:
            self.role_session = OnlineRoleInferenceSession.from_checkpoint(
                model_path=model_path,
                project_root=config.project_root,
            )
            self.logger.info(
                "Online position-role session loaded for tracking: %s",
                model_path,
            )
            if self.expected_roles_by_team:
                self.logger.info(
                    "Online position-role expected_roles_by_team activado: %s",
                    self.expected_roles_by_team,
                )
        except Exception as exc:
            self.logger.warning(
                "Se desactiva online special-seed role assignment: no se pudo cargar el checkpoint (%s)",
                exc,
            )
            self.enabled = False

    @staticmethod
    def _empty_observations_df():
        return pd.DataFrame(
            columns=[
                "match_id",
                "frame_id",
                "team_id",
                "player_id",
                "class_name",
                "x_m",
                "y_m",
                "x",
                "y",
                "visible",
                "confidence_tracking",
                "bbox",
                "role_label",
                "vx",
                "vy",
            ]
        )

    def _build_frame_observations(
        self,
        tracks,
        frame_id,
        *,
        include_special_ids=False,
        previous_positions=None,
    ):
        previous_positions = (
            self.prev_positions if previous_positions is None else previous_positions
        )
        rows = []
        frame_key_positions = {}
        for class_name in ("player", "goalkeeper"):
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                continue
            frame_tracks = class_frames[frame_id]
            if not isinstance(frame_tracks, dict):
                continue
            for track_id_raw, track_data in frame_tracks.items():
                if not isinstance(track_data, dict):
                    continue
                try:
                    player_id = int(track_id_raw)
                except (TypeError, ValueError):
                    continue
                is_special_id = int(player_id) in self.special_ids
                if is_special_id and not include_special_ids:
                    continue
                if track_data.get("synthetic_seed"):
                    continue
                team_id = track_data.get("team")
                if team_id is None:
                    continue
                field_position = _safe_field_position_m(track_data)
                if field_position is None:
                    continue
                x_m, y_m = field_position
                x = float(np.clip(x_m / 106.0, 0.0, 1.0))
                y = float(np.clip(y_m / 68.0, 0.0, 1.0))
                team_id_str = str(team_id)
                key = (team_id_str, int(player_id))
                previous = previous_positions.get(key)
                if previous is not None and frame_id > previous["frame_id"]:
                    dt = float(frame_id - previous["frame_id"])
                    vx = float((x - previous["x"]) / dt)
                    vy = float((y - previous["y"]) / dt)
                else:
                    vx = 0.0
                    vy = 0.0
                rows.append(
                    {
                        "match_id": self.video_path.stem.replace(" ", "_"),
                        "frame_id": int(frame_id),
                        "team_id": team_id_str,
                        "player_id": int(player_id),
                        "class_name": str(class_name),
                        "x_m": float(x_m),
                        "y_m": float(y_m),
                        "x": x,
                        "y": y,
                        "visible": 1,
                        "confidence_tracking": float(track_data.get("confidence", np.nan)),
                        "bbox": track_data.get("bbox"),
                        "role_label": pd.NA,
                        "vx": vx,
                        "vy": vy,
                    }
                )
                frame_key_positions[key] = {
                    "x": x,
                    "y": y,
                    "frame_id": int(frame_id),
                }

        if not rows:
            return self._empty_observations_df(), frame_key_positions
        return pd.DataFrame(rows), frame_key_positions

    @staticmethod
    def _set_track_role_payload(track_data, row):
        track_data["predicted_role_frame"] = str(row.predicted_role_frame)
        track_data["predicted_role_frame_confidence"] = float(
            row.predicted_role_frame_confidence
        )
        if hasattr(row, "predicted_role") and pd.notna(row.predicted_role):
            track_data["predicted_role"] = str(row.predicted_role)
        if hasattr(row, "predicted_role_confidence") and pd.notna(
            row.predicted_role_confidence
        ):
            track_data["predicted_role_confidence"] = float(
                row.predicted_role_confidence
            )
        if hasattr(row, "predicted_role_unconstrained") and pd.notna(
            row.predicted_role_unconstrained
        ):
            track_data["predicted_role_unconstrained"] = str(
                row.predicted_role_unconstrained
            )
        if hasattr(row, "predicted_role_confidence_unconstrained") and pd.notna(
            row.predicted_role_confidence_unconstrained
        ):
            track_data["predicted_role_confidence_unconstrained"] = float(
                row.predicted_role_confidence_unconstrained
            )
        if hasattr(row, "matched_model_role") and pd.notna(row.matched_model_role):
            track_data["matched_model_role"] = str(row.matched_model_role)
        if hasattr(row, "expected_role_slot") and pd.notna(row.expected_role_slot):
            track_data["expected_role_slot"] = str(row.expected_role_slot)
        if hasattr(row, "display_role_slot") and pd.notna(row.display_role_slot):
            track_data["display_role_slot"] = str(row.display_role_slot)
        if hasattr(row, "assignment_method") and pd.notna(row.assignment_method):
            track_data["assignment_method"] = str(row.assignment_method)
        if hasattr(row, "assignment_cost") and pd.notna(row.assignment_cost):
            track_data["assignment_cost"] = float(row.assignment_cost)

    def _record_raw_frame_predictions(self, frame_predictions_df):
        if frame_predictions_df.empty:
            return
        export_df = frame_predictions_df.copy()
        export_df["role_export_source"] = "online_model_frame"
        self.raw_frame_prediction_rows.extend(export_df.to_dict(orient="records"))

    def _record_greedy_diagnostics(
        self,
        *,
        frame_id,
        team_id,
        step_idx,
        pending_candidates,
        available_roles,
        chosen_track_id,
        chosen_raw_role,
    ):
        if not pending_candidates or not available_roles:
            return

        available_roles_list = [str(role) for role in available_roles]
        pending_track_ids = [int(track_id) for track_id in pending_candidates.keys()]
        for track_id, state in pending_candidates.items():
            observations = max(1, int(state.get("observations", 0)))
            x_mean = float(state.get("x_sum", 0.0)) / float(observations)
            y_mean = float(state.get("y_sum", 0.0)) / float(observations)
            for raw_role in available_roles_list:
                normalized_role = _normalize_role_token(raw_role)
                count = int(state.get("role_counts", {}).get(normalized_role, 0))
                confidence_sum = float(
                    state.get("confidence_sums", {}).get(normalized_role, 0.0)
                )
                mean_confidence = confidence_sum / max(1, count) if count > 0 else 0.0
                mean_probability = self._state_mean_prob_for_label(
                    state,
                    normalized_role,
                )
                priority = self._state_role_priority(state, normalized_role)
                self.greedy_diagnostic_rows.append(
                    {
                        "frame_id": int(frame_id),
                        "team_id": str(team_id),
                        "greedy_step": int(step_idx),
                        "player_id": int(track_id),
                        "candidate_slot": str(raw_role),
                        "candidate_role_label": str(normalized_role),
                        "observations": int(observations),
                        "x_mean": float(x_mean),
                        "y_mean": float(y_mean),
                        "role_count": int(count),
                        "role_mean_probability": float(mean_probability),
                        "role_mean_confidence": float(mean_confidence),
                        "role_confidence_sum": float(confidence_sum),
                        "priority_count": int(priority[0]),
                        "priority_mean_probability": float(priority[1]),
                        "priority_mean_confidence": float(priority[2]),
                        "priority_confidence_sum": float(priority[3]),
                        "pending_player_ids": "|".join(str(value) for value in pending_track_ids),
                        "available_slots": "|".join(available_roles_list),
                        "selected_assignment": bool(
                            int(track_id) == int(chosen_track_id)
                            and str(raw_role) == str(chosen_raw_role)
                        ),
                    }
                )

    @staticmethod
    def _tracks_player_role_summary_df(tracks):
        rows_by_key = {}
        for class_name, frames in tracks.items():
            if not isinstance(frames, list):
                continue
            for frame_id, frame_tracks in enumerate(frames):
                if not isinstance(frame_tracks, dict):
                    continue
                for track_id_raw, track_data in frame_tracks.items():
                    if not isinstance(track_data, dict):
                        continue
                    if (
                        track_data.get("predicted_role") is None
                        and track_data.get("predicted_role_frame") is None
                    ):
                        continue
                    try:
                        player_id = int(track_id_raw)
                    except (TypeError, ValueError):
                        player_id = str(track_id_raw)
                    key = (str(class_name), str(track_data.get("team")), str(player_id))
                    row = rows_by_key.setdefault(
                        key,
                        {
                            "class_name": str(class_name),
                            "team_id": None if track_data.get("team") is None else str(track_data.get("team")),
                            "player_id": player_id,
                            "frames_seen": 0,
                            "first_frame_id": int(frame_id),
                            "last_frame_id": int(frame_id),
                        },
                    )
                    row["frames_seen"] = int(row["frames_seen"]) + 1
                    row["last_frame_id"] = int(frame_id)
                    for col in (
                        "predicted_role",
                        "predicted_role_confidence",
                        "predicted_role_unconstrained",
                        "predicted_role_confidence_unconstrained",
                        "predicted_role_frame",
                        "predicted_role_frame_confidence",
                        "matched_model_role",
                        "expected_role_slot",
                        "display_role_slot",
                        "assignment_method",
                        "assignment_cost",
                        "stable_role_assignment_method",
                        "role_stabilized",
                        "role_stabilized_at_frame",
                        "role_stabilization_observations",
                    ):
                        if col in track_data and track_data.get(col) is not None:
                            row[col] = track_data.get(col)
        if not rows_by_key:
            return pd.DataFrame()
        player_df = pd.DataFrame(rows_by_key.values())
        sort_cols = [col for col in ("team_id", "player_id") if col in player_df.columns]
        if sort_cols:
            player_df = player_df.sort_values(sort_cols).reset_index(drop=True)
        return player_df

    def build_role_export_dataframes(self, tracks):
        if self.raw_frame_prediction_rows:
            frame_df = pd.DataFrame(self.raw_frame_prediction_rows)
            frame_sort_cols = [
                col
                for col in ("frame_id", "team_id", "player_id")
                if col in frame_df.columns
            ]
            if frame_sort_cols:
                frame_df = frame_df.sort_values(frame_sort_cols).reset_index(drop=True)
        else:
            frame_df = pd.DataFrame()
        player_df = self._tracks_player_role_summary_df(tracks)
        if self.greedy_diagnostic_rows:
            greedy_df = pd.DataFrame(self.greedy_diagnostic_rows)
            greedy_sort_cols = [
                col
                for col in ("frame_id", "team_id", "greedy_step", "player_id", "candidate_slot")
                if col in greedy_df.columns
            ]
            if greedy_sort_cols:
                greedy_df = greedy_df.sort_values(greedy_sort_cols).reset_index(drop=True)
        else:
            greedy_df = pd.DataFrame()
        return frame_df, player_df, greedy_df

    @staticmethod
    def _select_stable_role(role_counts, confidence_sums):
        best_role = None
        best_priority = None
        for role_name, count in role_counts.items():
            confidence_sum = float(confidence_sums.get(role_name, 0.0))
            mean_confidence = confidence_sum / max(1, int(count))
            priority = (int(count), mean_confidence, str(role_name))
            if best_priority is None or priority > best_priority:
                best_priority = priority
                best_role = str(role_name)
        return best_role

    def _role_labels_for_assignment(self):
        if self.role_session is None:
            return tuple()
        return tuple(
            str(label)
            for label in self.role_session.label_names
            if _normalize_role_token(label) != "POR"
        )

    @staticmethod
    def _state_mean_prob_for_label(state, label_name):
        prob_sums = state.get("prob_sums", {})
        observations = max(1, int(state.get("observations", 0)))
        return float(prob_sums.get(str(label_name), 0.0)) / float(observations)

    def _state_role_priority(self, state, role_label):
        role_label = str(role_label)
        observations = max(1, int(state.get("observations", 0)))
        count = int(state.get("role_counts", {}).get(role_label, 0))
        confidence_sum = float(state.get("confidence_sums", {}).get(role_label, 0.0))
        mean_confidence = confidence_sum / max(1, count) if count > 0 else 0.0
        mean_probability = self._state_mean_prob_for_label(state, role_label)
        return (
            int(count),
            float(mean_probability),
            float(mean_confidence),
            float(confidence_sum),
        )

    def _best_role_for_state(self, state, remaining_roles):
        ranked = []
        for raw_role in remaining_roles:
            normalized_role = _normalize_role_token(raw_role)
            ranked.append(
                (
                    self._state_role_priority(state, normalized_role),
                    str(raw_role),
                    normalized_role,
                )
            )
        if not ranked:
            return None
        ranked.sort(
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[0][2],
                item[0][3],
                item[2],
            ),
            reverse=True,
        )
        return ranked[0]

    def _state_is_ready_to_freeze(self, state):
        stable_role = self._select_stable_role(
            state.get("role_counts", {}),
            state.get("confidence_sums", {}),
        )
        stable_count = int(state.get("role_counts", {}).get(stable_role, 0))
        observations = int(state.get("observations", 0))
        stable_ratio = float(stable_count) / max(1, observations)
        if (
            observations >= self.role_stabilization_min_observations
            and stable_ratio >= self.role_stabilization_vote_ratio
        ):
            return True
        return observations >= self.role_stabilization_window_frames

    def _freeze_role_state(
        self,
        tracks,
        frame_id,
        track_id,
        role_label,
        confidence,
        expected_role_slot=None,
        assignment_method=None,
        increment_stats=True,
    ):
        state = self.role_state_by_track_id.get(int(track_id))
        if not isinstance(state, dict):
            return
        state["frozen_role"] = str(role_label)
        state["frozen_confidence"] = float(confidence)
        state["frozen_at_frame"] = int(frame_id)
        state["expected_role_slot"] = (
            None if expected_role_slot is None else str(expected_role_slot)
        )
        state["display_role_slot"] = (
            None if expected_role_slot is None else str(expected_role_slot)
        )
        state["assignment_method"] = (
            None if assignment_method is None else str(assignment_method)
        )
        if increment_stats:
            self.stats["stable_role_tracks_frozen"] += 1

    def _freeze_ready_roles_by_team(self, tracks, frame_id):
        role_labels = self._role_labels_for_assignment()
        if not role_labels:
            return

        if (
            self.expected_roles_by_team
            and self.role_stabilization_expected_roles_assignment == "ratio_priority"
        ):
            if self.role_stabilization_snapshot_done:
                return
            if int(frame_id) < int(self.role_stabilization_window_frames):
                return
            self._freeze_ready_roles_by_expected_roles_ratio_priority(
                tracks=tracks,
                frame_id=int(self.role_stabilization_window_frames),
                candidates_by_team=None,
            )
            self.role_stabilization_snapshot_done = True
            return

        candidates_by_team = {}
        for track_id, state in self.role_state_by_track_id.items():
            if not isinstance(state, dict):
                continue
            if state.get("frozen_role") is not None:
                continue
            if int(track_id) in self.special_ids:
                continue
            team_id = state.get("team_id")
            if team_id is None:
                continue
            if not self._state_is_ready_to_freeze(state):
                continue
            candidates_by_team.setdefault(str(team_id), []).append((int(track_id), state))

        handled_team_ids = set()
        if self.expected_roles_by_team:
            if self.role_stabilization_expected_roles_assignment == "greedy":
                handled_team_ids = self._freeze_ready_roles_by_expected_roles(
                    tracks=tracks,
                    frame_id=frame_id,
                    candidates_by_team=candidates_by_team,
                )
            elif self.role_stabilization_expected_roles_assignment == "ratio_priority":
                handled_team_ids = self._freeze_ready_roles_by_expected_roles_ratio_priority(
                    tracks=tracks,
                    frame_id=frame_id,
                    candidates_by_team=candidates_by_team,
                )
            else:
                handled_team_ids = self._freeze_ready_roles_by_expected_roles_hungarian(
                    tracks=tracks,
                    frame_id=frame_id,
                    candidates_by_team=candidates_by_team,
                )

        frozen_roles_by_team = {}
        for track_id, state in self.role_state_by_track_id.items():
            if not isinstance(state, dict):
                continue
            team_id = state.get("team_id")
            frozen_role = state.get("frozen_role")
            if team_id is None or frozen_role is None:
                continue
            frozen_roles_by_team.setdefault(str(team_id), set()).add(str(frozen_role))

        epsilon = 1e-9
        for team_id, candidates in candidates_by_team.items():
            if str(team_id) in handled_team_ids:
                continue
            used_roles = frozen_roles_by_team.get(str(team_id), set())
            available_roles = [
                role_label for role_label in role_labels if str(role_label) not in used_roles
            ]
            if not available_roles:
                continue

            row_track_ids = [int(track_id) for track_id, _ in candidates]
            cost_matrix = np.zeros((len(row_track_ids), len(available_roles)), dtype=np.float64)

            for row_pos, (_, state) in enumerate(candidates):
                for col_pos, role_label in enumerate(available_roles):
                    mean_prob = self._state_mean_prob_for_label(state, role_label)
                    cost_matrix[row_pos, col_pos] = -math.log(max(mean_prob, epsilon))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist()):
                track_id = row_track_ids[row_pos]
                role_label = str(available_roles[col_pos])
                mean_prob = math.exp(-float(cost_matrix[row_pos, col_pos]))
                self._freeze_role_state(
                    tracks,
                    frame_id=frame_id,
                    track_id=track_id,
                    role_label=role_label,
                    confidence=float(mean_prob),
                )
                frozen_roles_by_team.setdefault(str(team_id), set()).add(role_label)

        if self.expected_roles_by_team:
            self._resolve_duplicate_lateral_slots()

    def _remaining_expected_roles_for_team(
        self,
        team_id,
        treat_frozen_role_as_slot=True,
    ):
        if not self.expected_roles_by_team:
            return None
        expected_roles = self.expected_roles_by_team.get(str(team_id))
        if not expected_roles:
            return None

        used_slots = Counter()
        for state in self.role_state_by_track_id.values():
            if not isinstance(state, dict):
                continue
            if str(state.get("team_id")) != str(team_id):
                continue
            if state.get("frozen_role") is None:
                continue
            slot_label = state.get("expected_role_slot")
            if slot_label is None and treat_frozen_role_as_slot:
                slot_label = state.get("frozen_role")
            if slot_label is None:
                continue
            used_slots[_role_base_token(slot_label)] += 1

        remaining_roles = []
        for raw_role in expected_roles:
            normalized_role = _role_base_token(raw_role)
            if normalized_role == "POR":
                continue
            if used_slots.get(normalized_role, 0) > 0:
                used_slots[normalized_role] -= 1
                continue
            remaining_roles.append(str(raw_role))
        return remaining_roles

    def _resolve_duplicate_lateral_slots(self):
        if not self.expected_roles_by_team:
            return

        duplicate_bases = {"MC", "DC"}
        for team_id, expected_roles in self.expected_roles_by_team.items():
            role_counts = Counter(_role_base_token(role) for role in (expected_roles or []))
            active_duplicate_bases = [
                base_role
                for base_role in duplicate_bases
                if int(role_counts.get(base_role, 0)) >= 2
            ]
            if not active_duplicate_bases:
                continue

            team_states = [
                state
                for state in self.role_state_by_track_id.values()
                if isinstance(state, dict) and str(state.get("team_id")) == str(team_id)
            ]
            for base_role in active_duplicate_bases:
                candidates = []
                for state in team_states:
                    slot_label = state.get("expected_role_slot")
                    if slot_label is None:
                        continue
                    if _role_base_token(slot_label) != str(base_role):
                        continue
                    observations = max(1, int(state.get("lateral_observations", 0)))
                    mean_left = float(state.get("dist_left_sum", 0.0)) / float(observations)
                    mean_right = float(state.get("dist_right_sum", 0.0)) / float(observations)
                    candidates.append(
                        (
                            mean_left,
                            mean_right,
                            int(state.get("frozen_at_frame", 0) or 0),
                            state,
                        )
                    )

                if len(candidates) < 2:
                    for _, _, _, state in candidates:
                        state["display_role_slot"] = str(base_role)
                    continue

                candidates.sort(
                    key=lambda item: (
                        float(item[0]),
                        -float(item[1]),
                        int(item[2]),
                    )
                )
                left_state = candidates[0][3]
                right_state = candidates[-1][3]
                left_state["display_role_slot"] = f"{base_role}_IZQ"
                right_state["display_role_slot"] = f"{base_role}_DCHO"
                for _, _, _, state in candidates[1:-1]:
                    state["display_role_slot"] = str(base_role)

    def _complete_unassigned_expected_role_slots(self, tracks, frame_id):
        if not self.expected_roles_by_team:
            return

        for team_id, expected_roles in self.expected_roles_by_team.items():
            pending_candidates = {}
            available_roles = self._remaining_expected_roles_for_team(
                team_id,
                treat_frozen_role_as_slot=False,
            )
            if not available_roles:
                continue

            for track_id, state in self.role_state_by_track_id.items():
                if not isinstance(state, dict):
                    continue
                if state.get("frozen_role") is None:
                    continue
                if state.get("expected_role_slot") is not None:
                    continue
                if int(track_id) in self.special_ids:
                    continue
                if str(state.get("team_id")) != str(team_id):
                    continue
                if _normalize_role_token(state.get("frozen_role")) == "POR":
                    continue
                pending_candidates[int(track_id)] = state

            if not pending_candidates:
                continue

            ranked_expected_roles = _ordered_role_labels_from_keys(expected_roles or [])
            role_labels = _ordered_role_labels_from_keys(
                set(self._role_labels_for_assignment())
                | {
                    role
                    for state in pending_candidates.values()
                    for role in list(state.get("role_counts", {}).keys())
                    + list(state.get("prob_sums", {}).keys())
                }
                | set(ranked_expected_roles)
            )
            fill_assignments, _, _ = _resolve_remaining_snapshot_assignments(
                pending_candidates=pending_candidates,
                available_roles=available_roles,
                role_labels=role_labels,
                valid_roles=ranked_expected_roles,
                allow_zero_score=True,
                phase="fill_remaining",
            )
            for assignment in fill_assignments:
                state = self.role_state_by_track_id.get(int(assignment["player_id"]), {})
                mean_prob = 0.0
                if isinstance(state, dict):
                    mean_prob = self._state_mean_prob_for_label(
                        state,
                        str(assignment["slot_normalized"]),
                    )
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=int(assignment["player_id"]),
                    role_label=str(assignment["slot_normalized"]),
                    confidence=float(mean_prob),
                    expected_role_slot=str(assignment["slot"]),
                    assignment_method="ratio_priority_snapshot_fill_remaining_expected_roles",
                    increment_stats=False,
                )

    def _build_ready_role_predictions_df(self, candidates):
        if not candidates:
            return pd.DataFrame()

        role_labels = tuple(str(label) for label in self.role_session.label_names)
        rows = []
        for track_id, state in candidates:
            observations = max(1, int(state.get("observations", 0)))
            best_label = None
            best_prob = -1.0
            row = {
                "team_id": str(state.get("team_id")),
                "player_id": int(track_id),
                "class_name": "player",
                "frames_seen": observations,
                "x": float(state.get("x_sum", 0.0)) / float(observations),
                "y": float(state.get("y_sum", 0.0)) / float(observations),
            }
            for role_label in role_labels:
                mean_prob = self._state_mean_prob_for_label(state, role_label)
                row[f"prob_{role_label}"] = float(mean_prob)
                if role_label != "POR" and mean_prob > best_prob:
                    best_label = str(role_label)
                    best_prob = float(mean_prob)

            if best_label is None:
                continue
            row["predicted_role"] = best_label
            row["predicted_role_confidence"] = float(best_prob)
            rows.append(row)

        return pd.DataFrame(rows)

    def _freeze_ready_roles_by_expected_roles(
        self,
        tracks,
        frame_id,
        candidates_by_team,
    ):
        handled_team_ids = set()
        if not candidates_by_team:
            return handled_team_ids

        for team_id, candidates in candidates_by_team.items():
            remaining_roles = self._remaining_expected_roles_for_team(team_id)
            if not remaining_roles:
                continue
            handled_team_ids.add(str(team_id))

            pending_candidates = {
                int(track_id): state for track_id, state in candidates
            }
            available_roles = [str(role) for role in remaining_roles]
            greedy_step = 0

            while pending_candidates and available_roles:
                best_assignment = None
                for track_id, state in pending_candidates.items():
                    best_role = self._best_role_for_state(
                        state=state,
                        remaining_roles=available_roles,
                    )
                    if best_role is None:
                        continue
                    priority, raw_role, normalized_role = best_role
                    observations = int(state.get("observations", 0))
                    candidate = (
                        int(priority[0]),
                        float(priority[1]),
                        float(priority[2]),
                        float(priority[3]),
                        int(observations),
                        -int(track_id),
                    )
                    if best_assignment is None or candidate > best_assignment[0]:
                        best_assignment = (
                            candidate,
                            int(track_id),
                            str(raw_role),
                            str(normalized_role),
                            state,
                        )

                if best_assignment is None:
                    break

                _, track_id, raw_role, normalized_role, state = best_assignment
                self._record_greedy_diagnostics(
                    frame_id=frame_id,
                    team_id=team_id,
                    step_idx=greedy_step,
                    pending_candidates=pending_candidates,
                    available_roles=available_roles,
                    chosen_track_id=int(track_id),
                    chosen_raw_role=str(raw_role),
                )
                mean_prob = self._state_mean_prob_for_label(state, normalized_role)
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=int(track_id),
                    role_label=str(normalized_role),
                    confidence=float(mean_prob),
                    expected_role_slot=str(raw_role),
                    assignment_method="greedy_expected_roles_counts",
                )
                pending_candidates.pop(int(track_id), None)
                try:
                    available_roles.remove(str(raw_role))
                except ValueError:
                    pass
                greedy_step += 1
        return handled_team_ids

    def _freeze_ready_roles_by_expected_roles_hungarian(
        self,
        tracks,
        frame_id,
        candidates_by_team,
    ):
        handled_team_ids = set()
        if not candidates_by_team:
            return handled_team_ids

        epsilon = 1e-9
        for team_id, candidates in candidates_by_team.items():
            remaining_roles = self._remaining_expected_roles_for_team(team_id)
            if not remaining_roles:
                continue
            if len(remaining_roles) < len(candidates):
                continue

            handled_team_ids.add(str(team_id))
            row_track_ids = [int(track_id) for track_id, _ in candidates]
            available_roles = [str(role) for role in remaining_roles]
            cost_matrix = np.zeros(
                (len(row_track_ids), len(available_roles)),
                dtype=np.float64,
            )

            for row_pos, (_, state) in enumerate(candidates):
                for col_pos, raw_role in enumerate(available_roles):
                    normalized_role = _normalize_role_token(raw_role)
                    mean_prob = self._state_mean_prob_for_label(state, normalized_role)
                    cost_matrix[row_pos, col_pos] = -math.log(max(mean_prob, epsilon))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row_pos, col_pos in zip(row_ind.tolist(), col_ind.tolist()):
                track_id = row_track_ids[row_pos]
                raw_role = str(available_roles[col_pos])
                normalized_role = _normalize_role_token(raw_role)
                mean_prob = math.exp(-float(cost_matrix[row_pos, col_pos]))
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=track_id,
                    role_label=str(normalized_role),
                    confidence=float(mean_prob),
                    expected_role_slot=str(raw_role),
                    assignment_method="hungarian_expected_roles",
                )
        return handled_team_ids

    def _freeze_ready_roles_by_expected_roles_ratio_priority(
        self,
        tracks,
        frame_id,
        candidates_by_team,
    ):
        min_cumulative_ratio = float(self.role_stabilization_expected_roles_min_ratio)
        min_final_ratio = float(
            self.role_stabilization_expected_roles_min_final_ratio
        )
        min_count = int(self.role_stabilization_expected_roles_min_count)
        handled_team_ids = set()
        if not self.expected_roles_by_team:
            return handled_team_ids

        eligible_candidates_by_team = {}
        low_observation_candidates_by_team = {}
        for track_id, state in self.role_state_by_track_id.items():
            if not isinstance(state, dict):
                continue
            if state.get("frozen_role") is not None:
                continue
            if int(track_id) in self.special_ids:
                continue
            team_id = state.get("team_id")
            if team_id is None:
                continue
            observations = int(state.get("observations", 0))
            team_key = str(team_id)
            if observations >= min_count:
                eligible_candidates_by_team.setdefault(team_key, []).append(
                    (int(track_id), state)
                )
            else:
                low_observation_candidates_by_team.setdefault(team_key, []).append(
                    (int(track_id), state)
                )

        for team_id, candidates in eligible_candidates_by_team.items():
            remaining_roles = self._remaining_expected_roles_for_team(team_id)
            handled_team_ids.add(str(team_id))
            pending_candidates = {int(track_id): state for track_id, state in candidates}
            available_roles = [str(role) for role in (remaining_roles or [])]
            ranked_expected_roles = _ordered_role_labels_from_keys(
                self.expected_roles_by_team.get(str(team_id), [])
            )
            role_labels = _ordered_role_labels_from_keys(
                set(self._role_labels_for_assignment())
                | {
                    role
                    for _, state in candidates
                    for role in list(state.get("role_counts", {}).keys())
                    + list(state.get("prob_sums", {}).keys())
                }
            )

            while pending_candidates and available_roles:
                best_assignment = None
                for track_id, state in pending_candidates.items():
                    observations = max(1, int(state.get("observations", 0)))
                    valid_ranking = _build_valid_role_ranking_with_invalid_transfer(
                        state=state,
                        role_labels=role_labels,
                        valid_roles=ranked_expected_roles,
                        available_roles=available_roles,
                    )
                    if not valid_ranking:
                        continue

                    item = valid_ranking[0]
                    chosen_role = (
                        float(item["effective_ratio"]),
                        float(item["raw_ratio"]),
                        int(item["effective_count"]),
                        int(item["raw_count"]),
                        float(item["effective_mean_prob"]),
                        float(item["raw_mean_prob"]),
                        float(item["effective_mean_confidence"]),
                        float(item["raw_mean_confidence"]),
                        str(item["role"]),
                    )

                    if chosen_role is None:
                        continue

                    (
                        effective_ratio,
                        best_ratio,
                        effective_count,
                        best_count,
                        effective_prob,
                        best_prob,
                        effective_conf,
                        best_conf,
                        normalized_role,
                    ) = chosen_role
                    if (
                        effective_ratio < min_cumulative_ratio
                        or best_ratio < min_final_ratio
                    ):
                        continue

                    raw_role = next(
                        (
                            str(role_label)
                            for role_label in available_roles
                            if _normalize_role_token(role_label) == normalized_role
                        ),
                        None,
                    )
                    if raw_role is None:
                        continue

                    candidate = (
                        float(effective_ratio),
                        float(best_ratio),
                        int(effective_count),
                        int(best_count),
                        float(effective_prob),
                        float(best_prob),
                        float(effective_conf),
                        float(best_conf),
                        int(observations),
                        -int(track_id),
                    )
                    if best_assignment is None or candidate > best_assignment[0]:
                        best_assignment = (
                            candidate,
                            int(track_id),
                            str(raw_role),
                            str(normalized_role),
                            state,
                        )

                if best_assignment is None:
                    break

                _, track_id, raw_role, normalized_role, state = best_assignment
                mean_prob = self._state_mean_prob_for_label(state, normalized_role)
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=int(track_id),
                    role_label=str(normalized_role),
                    confidence=float(mean_prob),
                    expected_role_slot=str(raw_role),
                    assignment_method="ratio_priority_snapshot_expected_roles",
                )
                pending_candidates.pop(int(track_id), None)
                try:
                    available_roles.remove(str(raw_role))
                except ValueError:
                    pass

            residual_assignments, pending_candidates, available_roles = (
                _resolve_remaining_snapshot_assignments(
                    pending_candidates=pending_candidates,
                    available_roles=available_roles,
                    role_labels=role_labels,
                    valid_roles=ranked_expected_roles,
                )
            )
            for assignment in residual_assignments:
                state = self.role_state_by_track_id.get(int(assignment["player_id"]), {})
                mean_prob = 0.0
                if isinstance(state, dict):
                    mean_prob = self._state_mean_prob_for_label(
                        state,
                        str(assignment["slot_normalized"]),
                    )
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=int(assignment["player_id"]),
                    role_label=str(assignment["slot_normalized"]),
                    confidence=float(mean_prob),
                    expected_role_slot=str(assignment["slot"]),
                    assignment_method="ratio_priority_snapshot_remaining_expected_roles",
                )

            for track_id, state in pending_candidates.items():
                stable_role = self._select_stable_role(
                    state.get("role_counts", {}),
                    state.get("confidence_sums", {}),
                )
                if stable_role is None:
                    continue
                mean_prob = self._state_mean_prob_for_label(state, stable_role)
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=int(track_id),
                    role_label=str(stable_role),
                    confidence=float(mean_prob),
                    assignment_method="ratio_priority_snapshot_fallback",
                )

        for team_id, candidates in low_observation_candidates_by_team.items():
            handled_team_ids.add(str(team_id))
            for track_id, state in candidates:
                stable_role = self._select_stable_role(
                    state.get("role_counts", {}),
                    state.get("confidence_sums", {}),
                )
                if stable_role is None:
                    continue
                mean_prob = self._state_mean_prob_for_label(state, stable_role)
                self._freeze_role_state(
                    tracks=tracks,
                    frame_id=frame_id,
                    track_id=int(track_id),
                    role_label=str(stable_role),
                    confidence=float(mean_prob),
                    assignment_method="ratio_priority_snapshot_low_observations",
                )
        self._complete_unassigned_expected_role_slots(
            tracks=tracks,
            frame_id=frame_id,
        )
        self._resolve_duplicate_lateral_slots()
        return handled_team_ids

    def _update_role_state(self, tracks, frame_id, player_id, row):
        track_key = int(player_id)
        if hasattr(row, "predicted_role_unconstrained") and pd.notna(
            row.predicted_role_unconstrained
        ):
            label = str(row.predicted_role_unconstrained)
            confidence = float(
                getattr(
                    row,
                    "predicted_role_confidence_unconstrained",
                    row.predicted_role_frame_confidence,
                )
            )
        elif hasattr(row, "predicted_role") and pd.notna(row.predicted_role):
            label = str(row.predicted_role)
            confidence = float(
                getattr(
                    row,
                    "predicted_role_confidence",
                    row.predicted_role_frame_confidence,
                )
            )
        else:
            label = str(row.predicted_role_frame)
            confidence = float(row.predicted_role_frame_confidence)
        state = self.role_state_by_track_id.setdefault(
            track_key,
            {
                "observations": 0,
                "role_counts": {},
                "confidence_sums": {},
                "prob_sums": {},
                "team_id": None,
                "frozen_role": None,
                "frozen_confidence": None,
                "frozen_at_frame": None,
                "expected_role_slot": None,
                "display_role_slot": None,
                "assignment_method": None,
                "x_sum": 0.0,
                "y_sum": 0.0,
                "dist_left_sum": 0.0,
                "dist_right_sum": 0.0,
                "lateral_observations": 0,
            },
        )

        if state["frozen_role"] is not None:
            return state

        state["observations"] += 1
        state["team_id"] = str(getattr(row, "team_id", state.get("team_id")))
        state["role_counts"][label] = int(state["role_counts"].get(label, 0)) + 1
        state["confidence_sums"][label] = float(
            state["confidence_sums"].get(label, 0.0)
        ) + confidence
        state["x_sum"] = float(state.get("x_sum", 0.0)) + float(
            pd.to_numeric(getattr(row, "x", np.nan), errors="coerce")
            if not pd.isna(pd.to_numeric(getattr(row, "x", np.nan), errors="coerce"))
            else 0.0
        )
        state["y_sum"] = float(state.get("y_sum", 0.0)) + float(
            pd.to_numeric(getattr(row, "y", np.nan), errors="coerce")
            if not pd.isna(pd.to_numeric(getattr(row, "y", np.nan), errors="coerce"))
            else 0.0
        )
        dist_left_value = pd.to_numeric(
            getattr(row, "dist_left_sideline", np.nan),
            errors="coerce",
        )
        dist_right_value = pd.to_numeric(
            getattr(row, "dist_right_sideline", np.nan),
            errors="coerce",
        )
        if pd.notna(dist_left_value) and pd.notna(dist_right_value):
            state["dist_left_sum"] = float(state.get("dist_left_sum", 0.0)) + float(
                dist_left_value
            )
            state["dist_right_sum"] = float(state.get("dist_right_sum", 0.0)) + float(
                dist_right_value
            )
            state["lateral_observations"] = int(state.get("lateral_observations", 0)) + 1
        for role_label in self._role_labels_for_assignment():
            prob_col = f"prob_{role_label}"
            prob_value = float(getattr(row, prob_col, 0.0))
            state["prob_sums"][role_label] = float(
                state["prob_sums"].get(role_label, 0.0)
            ) + prob_value

        return state

    def _annotate_frame_with_roles(self, tracks, frame_id, frame_predictions_df):
        if frame_predictions_df.empty:
            return
        role_map = {}
        for row in frame_predictions_df.itertuples(index=False):
            role_map[(str(row.team_id), int(row.player_id))] = row

        for class_name in ("player", "goalkeeper"):
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                continue
            frame_tracks = class_frames[frame_id]
            if not isinstance(frame_tracks, dict):
                continue
            for track_id_raw, track_data in frame_tracks.items():
                if not isinstance(track_data, dict):
                    continue
                team_id = track_data.get("team")
                if team_id is None:
                    continue
                try:
                    player_id = int(track_id_raw)
                except (TypeError, ValueError):
                    continue
                row = role_map.get((str(team_id), int(player_id)))
                if row is None:
                    continue
                self._set_track_role_payload(track_data, row)
                state = self._update_role_state(
                    tracks,
                    frame_id=frame_id,
                    player_id=player_id,
                    row=row,
                )
                if state.get("frozen_role") is not None:
                    track_data["predicted_role_frame"] = str(state["frozen_role"])
                    track_data["predicted_role_frame_confidence"] = float(
                        state["frozen_confidence"]
                    )
                    track_data["predicted_role"] = str(state["frozen_role"])
                    track_data["predicted_role_confidence"] = float(
                        state["frozen_confidence"]
                    )
                    track_data["role_stabilized"] = True
                    track_data["role_stabilized_at_frame"] = int(
                        state["frozen_at_frame"]
                    )
                    track_data["role_stabilization_observations"] = int(
                        state["observations"]
                    )
                    if state.get("expected_role_slot") is not None:
                        track_data["expected_role_slot"] = str(
                            state["expected_role_slot"]
                        )
                    if state.get("display_role_slot") is not None:
                        track_data["display_role_slot"] = str(
                            state["display_role_slot"]
                        )
                    if state.get("assignment_method") is not None:
                        track_data["assignment_method"] = str(
                            state["assignment_method"]
                        )
                        track_data["stable_role_assignment_method"] = str(
                            state["assignment_method"]
                        )
                    else:
                        track_data["stable_role_assignment_method"] = (
                            "team_unique_hungarian"
                        )

    def _assign_special_seed_frame_teams(self, tracks, frame_id, frame_predictions_df):
        defender_candidates = []
        if not frame_predictions_df.empty:
            for row in frame_predictions_df.itertuples(index=False):
                predicted_role = (
                    getattr(row, "predicted_role", None)
                    or getattr(row, "predicted_role_frame", None)
                )
                if _normalize_role_token(predicted_role) not in self.defender_roles:
                    continue
                if pd.isna(getattr(row, "x_m", np.nan)) or pd.isna(
                    getattr(row, "y_m", np.nan)
                ):
                    continue
                defender_candidates.append(
                    {
                        "track_id": int(row.player_id),
                        "team": str(row.team_id),
                        "position": (float(row.x_m), float(row.y_m)),
                    }
                )

        frame_assignment_made = False
        for class_name in ("player", "goalkeeper"):
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                continue
            frame_tracks = class_frames[frame_id]
            if not isinstance(frame_tracks, dict):
                continue

            for special_id in self.special_ids:
                for track_key in _special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if not isinstance(track_data, dict):
                        continue
                    field_position = _safe_field_position_m(track_data)
                    if field_position is None:
                        continue

                    best_candidate = None
                    best_distance_m = None
                    for defender in defender_candidates:
                        distance_m = math.dist(field_position, defender["position"])
                        if best_distance_m is None or distance_m < best_distance_m:
                            best_distance_m = distance_m
                            best_candidate = defender

                    if best_candidate is not None:
                        resolved_team = str(best_candidate["team"])
                        track_data["team"] = resolved_team
                        track_data["special_team_assignment_source"] = "nearest_defender_role"
                        track_data["special_team_assignment_distance_m"] = float(
                            best_distance_m
                        )
                        track_data["special_team_assignment_track_id"] = int(
                            best_candidate["track_id"]
                        )
                        self.last_team_by_special_id[int(special_id)] = resolved_team
                        self.stats["special_seed_assigned_frames"] += 1
                        frame_assignment_made = True
                    elif int(special_id) in self.last_team_by_special_id:
                        track_data["team"] = self.last_team_by_special_id[int(special_id)]
                        track_data["special_team_assignment_source"] = (
                            "nearest_defender_role_carry"
                        )
                        track_data["special_team_assignment_distance_m"] = None
                        track_data["special_team_assignment_track_id"] = None
                        self.stats["special_seed_carry_frames"] += 1
                    else:
                        track_data["special_team_assignment_source"] = "unassigned"
                        track_data["special_team_assignment_distance_m"] = None
                        track_data["special_team_assignment_track_id"] = None
                        self.stats["special_seed_unassigned_frames"] += 1
        return frame_assignment_made

    def _annotate_special_goalkeeper_roles(self, tracks, frame_id):
        for class_name in ("player", "goalkeeper"):
            class_frames = tracks.get(class_name, [])
            if frame_id >= len(class_frames):
                continue
            frame_tracks = class_frames[frame_id]
            if not isinstance(frame_tracks, dict):
                continue

            for special_id in self.special_ids:
                for track_key in _special_seed_frame_track_keys(frame_tracks, special_id):
                    track_data = frame_tracks.get(track_key)
                    if not isinstance(track_data, dict):
                        continue
                    track_data["predicted_role_frame"] = "POR"
                    track_data["predicted_role_frame_confidence"] = 1.0
                    track_data["predicted_role"] = "POR"
                    track_data["predicted_role_confidence"] = 1.0
                    track_data["role_stabilized"] = True
                    track_data.setdefault("role_stabilized_at_frame", int(frame_id))
                    track_data.setdefault("role_stabilization_observations", 1)
                    track_data["stable_role_assignment_method"] = "manual_special_goalkeeper"

    def on_frame(self, tracks, frame_id):
        self.stats["processed_frames"] += 1
        if not self.enabled or self.role_session is None:
            return

        previous_positions_snapshot = dict(self.prev_positions)

        regular_observations, regular_positions = self._build_frame_observations(
            tracks,
            frame_id,
            include_special_ids=False,
            previous_positions=previous_positions_snapshot,
        )
        regular_frame_predictions_df = pd.DataFrame()

        if not regular_observations.empty:
            role_result = self.role_session.predict_frame(
                regular_observations,
                expected_roles_by_team=None,
            )
            regular_frame_predictions_df = role_result["frame_predictions_df"]
            self._record_raw_frame_predictions(regular_frame_predictions_df)

        self._assign_special_seed_frame_teams(
            tracks,
            frame_id,
            regular_frame_predictions_df,
        )

        self.stats["position_role_frame_predictions"] += int(
            len(regular_frame_predictions_df)
        )
        self.stats["position_role_player_predictions"] += int(
            regular_frame_predictions_df[
                ["team_id", "player_id"]
            ].drop_duplicates().shape[0]
            if not regular_frame_predictions_df.empty
            else 0
        )
        if not regular_frame_predictions_df.empty:
            self.stats["frames_with_role_predictions"] += 1
            self._annotate_frame_with_roles(
                tracks,
                frame_id,
                regular_frame_predictions_df,
            )
            self._freeze_ready_roles_by_team(tracks, frame_id)

        self._annotate_special_goalkeeper_roles(tracks, frame_id)

        for key, value in regular_positions.items():
            self.prev_positions[key] = value

    def summary(self):
        return dict(self.stats)


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
