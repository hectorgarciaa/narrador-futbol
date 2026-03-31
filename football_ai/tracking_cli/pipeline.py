import json
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from football_ai.core import Logger, get_config, get_logger
from football_ai.evaluation import Evaluator
from football_ai.positions import (
    LineupSlotMatcher,
    OnlineSpecialSeedRoleAssigner,
    build_role_artifacts_output_dir,
    build_expected_roles_by_team,
    build_role_predictions_output_paths,
    build_team_colors_by_team,
    copy_output_artifact,
    load_lineup_spec,
    save_dataframe_csv,
    save_role_visualizations,
)
from football_ai.tracking import Tracker
from football_ai.visualization import Drawer

from .paths import (
    build_output_video_path,
    build_debug_frames_output_path,
    build_tracking_metrics_output_paths,
    build_tracks_output_paths,
    resolve_video_path,
)
from .persistence import save_debug_frames, save_result, save_summary, upsert_tracking_metrics_dataset

COLOR_NAME_TO_RGB = {
    "white": (255, 255, 255),
    "blanco": (255, 255, 255),
    "black": (0, 0, 0),
    "negro": (0, 0, 0),
    "red": (255, 0, 0),
    "rojo": (255, 0, 0),
    "blue": (0, 102, 255),
    "azul": (0, 102, 255),
    "green": (0, 170, 0),
    "verde": (0, 170, 0),
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


def _normalize_token(value):
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.strip().lower()
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized


def _parse_rgb_triplet(raw_value):
    numbers = re.findall(r"\d+", str(raw_value))
    if len(numbers) != 3:
        return None
    values = tuple(int(v) for v in numbers)
    if any(v < 0 or v > 255 for v in values):
        raise ValueError(f"RGB invalido '{raw_value}'. Cada canal debe estar en 0..255.")
    return values


def _parse_color_to_rgb(raw_color):
    color_text = str(raw_color).strip().strip('"').strip("'")
    normalized_color = _normalize_token(color_text)
    if normalized_color in COLOR_NAME_TO_RGB:
        return COLOR_NAME_TO_RGB[normalized_color]

    hex_match = re.fullmatch(r"#?([0-9a-fA-F]{6})", color_text)
    if hex_match:
        hex_code = hex_match.group(1)
        return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))

    rgb_triplet = _parse_rgb_triplet(color_text)
    if rgb_triplet is not None:
        return rgb_triplet

    raise ValueError(
        f"Color no soportado '{raw_color}'. Usa nombre comun, HEX (#RRGGBB) o RGB (255,255,255)."
    )


def _rgb_to_lab_opencv(rgb_color):
    rgb_pixel = np.array([[list(rgb_color)]], dtype=np.uint8)
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)[0, 0]
    return lab_pixel.astype(np.float32)


def _parse_team_color_overrides(raw_text):
    text = str(raw_text or "").strip()
    if not text:
        return []
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    if not text.strip():
        return []

    overrides = []
    for item in text.split(","):
        segment = item.strip()
        if not segment:
            continue
        separator = ":" if ":" in segment else "=" if "=" in segment else None
        if separator is None:
            raise ValueError(f"Formato invalido '{segment}'. Usa 'Equipo:color'.")
        team_name, color_value = segment.split(separator, 1)
        team_name = team_name.strip().strip('"').strip("'")
        color_value = color_value.strip().strip('"').strip("'")
        if not team_name or not color_value:
            raise ValueError(f"Par invalido '{segment}'.")
        overrides.append((team_name, color_value))
    return overrides


def _build_team_colors_from_raw_mapping(raw_team_colors):
    colors = {}
    for team_name, raw_color in (raw_team_colors or {}).items():
        colors[str(team_name)] = _rgb_to_lab_opencv(_parse_color_to_rgb(raw_color))
    return colors


def _apply_team_color_overrides(base_team_colors, raw_overrides, logger):
    updated = dict(base_team_colors)
    applied = {}
    for team_name, raw_color in _parse_team_color_overrides(raw_overrides):
        rgb = _parse_color_to_rgb(raw_color)
        lab = _rgb_to_lab_opencv(rgb)
        updated[str(team_name)] = lab
        applied[str(team_name)] = {"rgb": list(rgb), "lab_opencv": [float(v) for v in lab]}
    if applied:
        logger.info("Team colors override aplicado: %s", json.dumps(applied, ensure_ascii=False))
    return updated


def _normalize_team_detector_runtime_conf(team_detector_conf, team_mode=None):
    runtime_conf = dict(team_detector_conf or {})
    runtime_conf.pop("with_ref", None)

    normalized_team_mode = None
    if team_mode is not None:
        normalized_team_mode = str(team_mode).strip().lower()

    if normalized_team_mode == "auto-bootstrap":
        runtime_conf.pop("team_colors", None)
    elif not runtime_conf.get("team_colors"):
        runtime_conf.pop("team_colors", None)

    return runtime_conf


def _resolve_runtime_device_label(raw_device):
    normalized = str(raw_device or "cpu").strip().lower()
    if normalized.startswith("cuda"):
        return f"{normalized} (cuda gpu)"
    return f"{normalized} (cpu)"


def _resolve_pnlcalib_device(tracker):
    field_projector = getattr(tracker, "field_projector", None)
    if field_projector is None:
        return "cpu (pnlcalib disabled)"
    estimator = getattr(field_projector, "estimator", None)
    runtime = getattr(estimator, "runtime", None)
    runtime_device = getattr(runtime, "device", "cpu")
    return _resolve_runtime_device_label(runtime_device)


def _resolve_frame_hook_device(frame_hook):
    if frame_hook is None:
        return "cpu (frame_hook disabled)"
    hook_owner = getattr(frame_hook, "__self__", None)
    role_session = getattr(hook_owner, "role_session", None)
    if role_session is None:
        return "cpu (frame_hook sin backend torch)"
    runtime_device = getattr(role_session, "device", "cpu")
    return _resolve_runtime_device_label(runtime_device)


def run_tracking_pipeline(args):
    # Load configuration
    config = get_config()

    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)

    logger.info("Starting football tracking system")

    try:
        lineup_spec = None
        lineup_matcher = None
        lineup_expected_roles_by_team = None
        lineup_team_colors_raw = {}
        if getattr(args, "lineup_spec", None):
            lineup_spec = load_lineup_spec(
                args.lineup_spec,
                project_root=config.project_root,
            )
            lineup_matcher = LineupSlotMatcher(lineup_spec)
            lineup_expected_roles_by_team = build_expected_roles_by_team(lineup_spec)
            lineup_team_colors_raw = build_team_colors_by_team(lineup_spec)

        # Get paths and parameters from config
        model_path = str(config.get_path("paths", "models", "modelo_base"))
        effective_video_shortcut = args.video_shortcut or (
            str(lineup_spec.get("video_source") or "").strip()
            if lineup_spec is not None
            else None
        )
        video_path, video_source = resolve_video_path(config, effective_video_shortcut)
        
        output = build_output_video_path(config, video_path)
        output_path_named, output_path_legacy = build_tracks_output_paths(config, video_path)
        summary_path, metrics_dataset_path = build_tracking_metrics_output_paths(config, video_path)
        
        role_artifacts_dir = build_role_artifacts_output_dir(config, video_path)
        role_frame_csv_path, role_player_csv_path, role_greedy_csv_path = build_role_predictions_output_paths(config, video_path, use_artifacts_dir=True)
        role_frame_csv_path_legacy, role_player_csv_path_legacy, role_greedy_csv_path_legacy = build_role_predictions_output_paths(config, video_path, use_artifacts_dir=False)

        # Configuration parameters
        visualization_conf = config.visualization
        show_kmeans = config.get("visualization", "show_kmeans")
        show_output = config.get("visualization", "show_output")
        four_panel_enabled = bool(visualization_conf.get("four_panel_enabled", False))

        # Tracking configuration
        detector_conf = config.detection
        team_detector_conf = dict(config.team_detector)
        projector_conf = config.projector
        bytetracker_conf = config.bytetracker
        ball_conf = config.ball

        tracker_conf = config.tracking

        if lineup_spec is not None:
            lineup_colors = _build_team_colors_from_raw_mapping(lineup_team_colors_raw)
            team_detector_conf["team_colors"] = {
                team: [float(channel) for channel in color]
                for team, color in lineup_colors.items()
            }

        if getattr(args, "team_colors", None):
            base_colors = {
                team_name: np.asarray(color, dtype=np.float32)
                for team_name, color in dict(team_detector_conf.get("team_colors", {})).items()
            }
            override_colors = _apply_team_color_overrides(base_colors, args.team_colors, logger)
            team_detector_conf["team_colors"] = {
                team_name: [float(channel) for channel in color]
                for team_name, color in override_colors.items()
            }

        team_mode = getattr(args, "team_mode", None)

        if getattr(args, "team_bootstrap_min_samples", None) is not None:
            team_detector_conf["min_samples"] = int(args.team_bootstrap_min_samples)
        if getattr(args, "team_bootstrap_min_cluster_samples", None) is not None:
            team_detector_conf["min_size_cluster"] = int(args.team_bootstrap_min_cluster_samples)

        team_detector_conf = _normalize_team_detector_runtime_conf(
            team_detector_conf,
            team_mode=team_mode,
        )

        logger.info(f"Model: {model_path}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Video source: {video_source}")
        logger.info(f"Named tracks JSON output: {output_path_named}")
        logger.info(f"Legacy tracks JSON output: {output_path_legacy}")
        logger.info(f"Summary JSON output: {summary_path}")
        logger.info(f"Role artifacts directory: {role_artifacts_dir}")
        logger.info(f"Frame role CSV output: {role_frame_csv_path}")
        logger.info(f"Player role CSV output: {role_player_csv_path}")
        logger.info(f"Greedy role CSV output: {role_greedy_csv_path}")
        logger.info(f"Tracking metrics dataset CSV: {metrics_dataset_path}")
        logger.info(f"Tracking configuration: {tracker_conf}")
        logger.info(f"Field tracking configuration: {projector_conf}")
        logger.info(f"Team detector configuration: {team_detector_conf}")

        # Run tracking
        tracker = Tracker(model_path, detector_conf, team_detector_conf, bytetracker_conf,
                          ball_conf, tracker_conf, projector_conf, config.project_root)
        
        online_special_seed_role_assigner = OnlineSpecialSeedRoleAssigner(
            config,
            video_path,
            logger,
            expected_roles_by_team_override=lineup_expected_roles_by_team,
            lineup_matcher=lineup_matcher,
        )
        frame_hook = online_special_seed_role_assigner.on_frame

        if bool(tracker_conf.get("print_runtime_devices", True)):
            print(
                f"[runtime] pnlcalib: {_resolve_pnlcalib_device(tracker)}",
                flush=True,
            )
            print(
                f"[runtime] frame_hook: {_resolve_frame_hook_device(frame_hook)}",
                flush=True,
            )

        profile_phases_enabled = bool(getattr(args, "profile_phases", False))
        if not profile_phases_enabled:
            profile_phases_enabled = bool(tracker_conf.get("profile_phases", False))
        logger.info("Phase profiling per frame: %s", profile_phases_enabled)

        logger.info("Extracting tracks from video...")
        
        tracks = tracker.get_tracks(
            video_path,
            show_kmeans,
            frame_hook=frame_hook,
            collect_visual_debug=four_panel_enabled,
            profile_phases=profile_phases_enabled,
        )

        if four_panel_enabled:
            debug_frames_path = build_debug_frames_output_path(config, video_path)
            save_debug_frames(tracker.visualization_debug_frames, debug_frames_path, logger)
        
        role_postprocess_result = online_special_seed_role_assigner.summary() if online_special_seed_role_assigner.enabled else None
        if role_postprocess_result is not None:
            logger.info(
                "Online position-role assignment applied: %s player predictions, %s frame predictions",
                role_postprocess_result.get("position_role_player_predictions"),
                role_postprocess_result.get("position_role_frame_predictions"),
            )

        # Draw tracks
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors, visualization_conf=visualization_conf)
        logger.info("Drawing tracks on video...")
        drawer.draw_tracks(
            tracks,
            video_path,
            output,
            show=show_output,
            four_panel=four_panel_enabled,
            debug_frames=(tracker.visualization_debug_frames if four_panel_enabled else None),
            expected_counts=tracker.max_tracks_per_class,
        )
        logger.info(f"Video with tracks saved to: {output}")

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
        save_result(tracks, output_path_named, logger)
        if output_path_legacy != output_path_named:
            save_result(tracks, output_path_legacy, logger)
        if online_special_seed_role_assigner.enabled:
            role_frame_df, role_player_df, role_greedy_df = online_special_seed_role_assigner.build_role_export_dataframes(tracks)
            save_dataframe_csv(role_frame_df, role_frame_csv_path, logger, "Frame role predictions CSV")
            copy_output_artifact(role_frame_csv_path, role_frame_csv_path_legacy, logger, "Frame role predictions CSV")
            save_dataframe_csv(role_player_df, role_player_csv_path, logger, "Player role summary CSV")
            copy_output_artifact(role_player_csv_path, role_player_csv_path_legacy, logger, "Player role summary CSV")
            save_dataframe_csv(role_greedy_df, role_greedy_csv_path, logger, "Greedy role diagnostics CSV")
            copy_output_artifact(role_greedy_csv_path, role_greedy_csv_path_legacy, logger, "Greedy role diagnostics CSV")
            save_role_visualizations(
                frame_df=role_frame_df,
                player_df=role_player_df,
                config=config,
                video_path=video_path,
                output_dir=role_artifacts_dir,
                expected_roles_by_team=online_special_seed_role_assigner.expected_roles_by_team,
                assignment_method=online_special_seed_role_assigner.role_stabilization_expected_roles_assignment,
                min_count=online_special_seed_role_assigner.role_stabilization_expected_roles_min_count,
                min_cumulative_ratio=online_special_seed_role_assigner.role_stabilization_expected_roles_min_ratio,
                min_final_ratio=online_special_seed_role_assigner.role_stabilization_expected_roles_min_final_ratio,
                logger=logger,
                legacy_output_dir=Path(role_frame_csv_path_legacy).parent,
            )
        save_summary(summary, summary_path, logger)
        upsert_tracking_metrics_dataset(metrics_dataset_path, video_path, video_source, output_path_named, summary_path, summary, logger)

        # Show summary
        logger.info("Evaluation summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        print("\n=== TRACKING SUMMARY ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1

    logger.info("Process completed successfully")
    return 0
