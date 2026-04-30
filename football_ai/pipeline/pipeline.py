import json
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_ai.core import Logger, get_config, get_logger
from football_ai.evaluation import Evaluator
from football_ai.positions import (
    LineupSlotMatcher,
    build_role_artifacts_output_dir,
    build_expected_roles_by_team,
    build_role_predictions_output_paths,
    build_team_colors_by_team,
    copy_output_artifact,
    load_lineup_spec,
    save_dataframe_csv,
)
from football_ai.tracking import Tracker
from football_ai.visualization import Drawer
from football_ai.visualization.pathcrf_drawer import PathCRFDrawer
from football_ai.actions_incremental import ActionsDetector, ActionsDetectorConfig

from .paths import (
    build_output_video_path,
    build_debug_frames_output_path,
    build_tracking_metrics_output_paths,
    build_tracks_output_paths,
    resolve_video_path,
)
from .live_commentary import compose_frame_hooks, create_app_live_commentary_bridge_from_env
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


def _resolve_pnlcalib_device(tracking_phase):
    field_projector = getattr(tracking_phase.tracker, "field_projector", None) if hasattr(tracking_phase, "tracker") else None
    if field_projector is None:
        return "cpu (pnlcalib disabled)"
    runtime = getattr(field_projector, "runtime", None)
    runtime_device = getattr(runtime, "device", "cpu")
    return _resolve_runtime_device_label(runtime_device)


def _resolve_frame_hook_device(frame_hook):
    if frame_hook is None:
        return "cpu (frame_hook disabled)"
    hook_owner = getattr(frame_hook, "__self__", None)
    role_session = getattr(hook_owner, "role_session", None)
    if role_session is None:
        role_session = getattr(frame_hook, "role_session", None)
    if role_session is None:
        return "cpu (frame_hook sin backend torch)"
    runtime_device = getattr(role_session, "device", "cpu")
    return _resolve_runtime_device_label(runtime_device)


def _resolve_position_infering_device(position_phase):
    if position_phase is None:
        return "cpu (position infering disabled)"
    role_session = getattr(position_phase, "role_session", None)
    if role_session is None:
        return "cpu (position infering sin backend torch)"
    runtime_device = getattr(role_session, "device", "cpu")
    return _resolve_runtime_device_label(runtime_device)

def resolve_lineup_spec(args, config):
    lineup_spec = getattr(args, "lineup_spec", None)
    if not lineup_spec:
        lineup_spec = config.get("tracking", "lineup_spec", default=None)
    lineup_matcher = None
    lineup_expected_roles_by_team = None
    lineup_team_colors_raw = {}

    if lineup_spec:
        lineup_spec = load_lineup_spec(lineup_spec, project_root=config.project_root)
        lineup_matcher = LineupSlotMatcher(lineup_spec)
        lineup_expected_roles_by_team = build_expected_roles_by_team(lineup_spec)
        lineup_team_colors_raw = build_team_colors_by_team(lineup_spec)
    
    return lineup_spec, lineup_matcher, lineup_expected_roles_by_team, lineup_team_colors_raw


def run_tracking_pipeline(args):
    # Load configuration
    config = get_config()

    # Configure logging
    Logger.setup_from_config(config)
    logger = get_logger(__name__)

    logger.info("Starting football tracking system")

    try:
        (
            lineup_spec,
            lineup_matcher,
            lineup_expected_roles_by_team,
            lineup_team_colors_raw
        ) = resolve_lineup_spec(args, config)

        # Get paths and parameters from config
        model_path = (
            str(Path(args.model_path).expanduser().resolve())
            if getattr(args, "model_path", None)
            else config.get_path("paths", "models", "modelo_base")
        )
        effective_video_shortcut = args.video_shortcut or (
            str(lineup_spec.get("video_source") or "").strip()
            if lineup_spec is not None
            else None
        )
        video_path, video_source = resolve_video_path(config, effective_video_shortcut)

        output_root_arg = getattr(args, "output_root", None)
        if output_root_arg:
            output_root = Path(output_root_arg).expanduser()
            if not output_root.is_absolute():
                output_root = (config.project_root / output_root).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            output = str(output_root / "tracking.mp4")
            output_path_named = str(output_root / "tracks.json")
            output_path_legacy = str(output_root / "tracks_legacy.json")
            summary_path = str(output_root / "summary.json")
            metrics_dataset_path = str(output_root / "tracking_metrics.csv")
        else:
            output = build_output_video_path(config, video_path)
            output_path_named, output_path_legacy = build_tracks_output_paths(config, video_path)
            summary_path, metrics_dataset_path = build_tracking_metrics_output_paths(config, video_path)
        
        role_artifacts_dir = build_role_artifacts_output_dir(config, video_path)
        role_frame_csv_path, role_player_csv_path, role_greedy_csv_path = build_role_predictions_output_paths(config, video_path, use_artifacts_dir=True)
        role_frame_csv_path_legacy, role_player_csv_path_legacy, role_greedy_csv_path_legacy = build_role_predictions_output_paths(config, video_path, use_artifacts_dir=False)

        # Configuration parameters
        visualization_conf = dict(config.visualization or {})
        show_kmeans = config.get("visualization", "show_kmeans")
        show_output = config.get("visualization", "show_output")
        four_panel_enabled = bool(visualization_conf.get("four_panel_enabled", False))
        if bool(getattr(args, "force_four_panel_debug", False)):
            four_panel_enabled = True

        # Tracking configuration
        detector_conf = config.detection
        team_detector_conf = dict(config.team_detector)
        projector_conf = config.projector
        bytetracker_conf = config.bytetracker
        ball_conf = config.ball

        tracker_conf = config.tracking

        if lineup_spec is not None:
            lineup_colors = _build_team_colors_from_raw_mapping(lineup_team_colors_raw)
            team_detector_conf.setdefault("team_color_model_conf", {})
            team_detector_conf["team_color_model_conf"]["team_colors"] = {
                team: [float(channel) for channel in color]
                for team, color in lineup_colors.items()
            }

        if getattr(args, "team_colors", None):
            base_colors = {
                team_name: np.asarray(color, dtype=np.float32)
                for team_name, color in dict(
                    team_detector_conf.get("team_color_model_conf", {}).get("team_colors", {})
                ).items()
            }
            override_colors = _apply_team_color_overrides(base_colors, args.team_colors, logger)
            team_detector_conf.setdefault("team_color_model_conf", {})
            team_detector_conf["team_color_model_conf"]["team_colors"] = {
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
        experiment_label = str(
            getattr(args, "experiment_label", None) or video_source
        )
        logger.info(f"Video source: {video_source}")
        logger.info(f"Experiment label: {experiment_label}")
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
        from football_ai.tracking import TrackingPhase
        from football_ai.posession.phase import PosessionPhase
        from football_ai.positions import PositionInferingPhase
        from football_ai.actions_incremental import ActionsDetectorConfig, ActionsDetectorPhase, ActionsRuntimeConfig
        from scripts.utils import iter_video_frames

        tracking_phase = TrackingPhase(
            model_path,
            detector_conf,
            team_detector_conf,
            bytetracker_conf,
            ball_conf,
            tracker_conf,
            projector_conf,
            config.project_root,
        )
        posession_phase = PosessionPhase(tracker_conf.get("possession"))
        
        position_phase = None
        actions_phase = None
        if config is not None and video_path is not None and logger is not None:
            position_phase = PositionInferingPhase(
                config,
                video_path,
                logger,
                expected_roles_by_team_override=lineup_expected_roles_by_team,
                lineup_matcher=lineup_matcher,
            )
        actions_conf = dict(tracker_conf.get("actions") or {})
        if bool(actions_conf.get("enabled", True)):
            actions_phase = ActionsDetectorPhase(
                ActionsRuntimeConfig(
                    detector=ActionsDetectorConfig(
                        fps=float(actions_conf.get("fps", 25.0)),
                        window_size_frames=(
                            int(actions_conf["window_size_frames"])
                            if actions_conf.get("window_size_frames") is not None
                            else None
                        ),
                    ),
                    cadence_frames=max(1, int(actions_conf.get("cadence_frames", 10))),
                    min_frames_warmup=max(1, int(actions_conf.get("min_frames_warmup", 50))),
                    min_event_duration=max(1, int(actions_conf.get("min_event_duration", 10))),
                    window_seconds=actions_conf.get("window_seconds"),
                    sample_freq=actions_conf.get("sample_freq"),
                )
            )

        online_commentary_bridge = create_app_live_commentary_bridge_from_env()
        if online_commentary_bridge is not None:
            logger.info("Actions live commentary bridge activado para esta ejecución.")
        frame_hook = compose_frame_hooks(
            (online_commentary_bridge.on_frame if online_commentary_bridge is not None else None),
        )

        if bool(tracker_conf.get("print_runtime_devices", True)):
            print(
                f"[runtime] pnlcalib: {_resolve_pnlcalib_device(tracking_phase)}",
                flush=True,
            )
            print(
                f"[runtime] position_infering: {_resolve_position_infering_device(position_phase)}",
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
        
        tracks = {
            "player": [],
            "goalkeeper": [],
            "referee": [],
            "ball": [],
            "possession": [],
            "actions_incremental": [],
        }
        visual_debug_frames = []

        tracking_phase.reset()
        posession_phase.reset()
        if position_phase:
            position_phase.reset()
        if actions_phase:
            actions_phase.reset()

        for frame_index, frame_time_ms, frame_bgr in iter_video_frames(video_path, getattr(args, "max_frames", None)):
            tracking_packet, tracking_ms = tracking_phase.process(
                frame_bgr,
                frame_index=frame_index,
                frame_time_ms=frame_time_ms,
                show_kmeans=show_kmeans,
                collect_visual_debug=four_panel_enabled,
            )
            
            posession_packet, posession_ms = posession_phase.process(tracking_packet)
            
            final_packet = posession_packet
            position_ms = 0.0
            actions_ms = 0.0
            if position_phase:
                position_packet, position_ms = position_phase.process(posession_packet)
                final_packet = position_packet
            if actions_phase:
                actions_packet, actions_ms = actions_phase.process(final_packet)
                final_packet = actions_packet
            
            tracks_frame = final_packet["clean"]["tracks_frame"]
            for cls in ["player", "goalkeeper", "referee", "ball"]:
                tracks[cls].append(dict(tracks_frame[cls]))
            
            possession_info = final_packet["clean"].get("possession", {})
            tracks["possession"].append(dict(possession_info))
            actions_info = final_packet["clean"].get("actions_incremental", {})
            actions_trace = final_packet["trace"].get("actions_incremental", {})
            tracks["actions_incremental"].append(
                {
                    "raw_edge": dict(actions_info.get("raw_edge") or {}) if isinstance(actions_info, dict) else {},
                    "confirmed_action": dict(actions_info.get("confirmed_action") or {}) if isinstance(actions_info, dict) else {},
                    "action_metadata": dict(actions_info.get("action_metadata") or {}) if isinstance(actions_info, dict) else {},
                    "person_slot_assignments": dict(actions_trace.get("person_slot_assignments") or {}) if isinstance(actions_trace, dict) else {},
                    "referee_slot_assignments": dict(actions_trace.get("referee_slot_assignments") or {}) if isinstance(actions_trace, dict) else {},
                }
            )

            if four_panel_enabled:
                trace_debug = final_packet["trace"].get("visual_debug")
                if trace_debug:
                    visual_debug_frames.append(trace_debug)

            if profile_phases_enabled:
                prof = tracking_packet["trace"].get("profile_ms", {})
                d_ms = prof.get("detector_ms", 0.0)
                p_ms = prof.get("proj_ms", 0.0)
                f_ms = prof.get("filter_ms", 0.0)
                i_ms = prof.get("id_ms", 0.0)
                b_ms = prof.get("byte_ms", 0.0)
                c_ms = prof.get("canon_ms", 0.0)
                
                total_ms = tracking_ms + posession_ms + position_ms
                total_ms += actions_ms
                pos_inf = f" | PosInf: {position_ms:.1f}ms" if position_phase else ""
                act_inf = f" | Actions: {actions_ms:.1f}ms" if actions_phase else ""
                
                print(
                    f"[frame {frame_index:04d}] "
                    f"Detect: {d_ms:.1f}ms | Proj: {p_ms:.1f}ms | Filter: {f_ms:.1f}ms | "
                    f"Id: {i_ms:.1f}ms | Byte: {b_ms:.1f}ms | Canon: {c_ms:.1f}ms | "
                    f"Poss: {posession_ms:.1f}ms{pos_inf}{act_inf} || Total AI: {total_ms:.1f}ms",
                    flush=True
                )

            if frame_hook is not None:
                frame_hook(frame_index, frame_bgr, final_packet)

        # Config required max tracks for drawing
        max_tracks_per_class = {
            "player": 22,
            "goalkeeper": 2,
            "referee": 3,
            "ball": 1
        }

        if four_panel_enabled and visual_debug_frames:
            debug_frames_path = build_debug_frames_output_path(config, video_path)
            save_debug_frames(visual_debug_frames, debug_frames_path, logger)
        
        role_postprocess_result = (
            position_phase.summary()
            if position_phase is not None and getattr(position_phase, "enabled", True)
            else None
        )
        if role_postprocess_result is not None:
            logger.info(
                "Online position-role assignment applied: %s player predictions, %s frame predictions",
                role_postprocess_result.get("position_role_player_predictions"),
                role_postprocess_result.get("position_role_frame_predictions"),
            )

        # Draw tracks
        drawer_visualization_conf = dict(visualization_conf or {})
        runtime_team_colors = None
        try:
            runtime_team_colors = (
                tracking_phase.tracker
                .identification_phase
                .team_detector
                .color_model
                .team_colors
            )
        except Exception:
            runtime_team_colors = (
                team_detector_conf.get("team_color_model_conf", {}).get("team_colors")
            )
        if isinstance(runtime_team_colors, dict) and runtime_team_colors:
            drawer_visualization_conf["team_colors"] = {
                str(team_name): [float(channel) for channel in color]
                for team_name, color in runtime_team_colors.items()
                if color is not None
            }
            drawer_visualization_conf["team_colors_space"] = "lab_opencv"
        colors = config.get_visualization_colors()
        drawer = Drawer(colors=colors, visualization_conf=drawer_visualization_conf)
        logger.info("Drawing tracks on video...")
        if not bool(getattr(args, "skip_render_video", False)):
            drawer.draw_tracks(
                tracks,
                video_path,
                output,
                show=show_output,
                four_panel=four_panel_enabled,
                debug_frames=(visual_debug_frames if four_panel_enabled else None),
                expected_counts=max_tracks_per_class,
            )
            logger.info(f"Video with tracks saved to: {output}")
        else:
            logger.info("Skipping annotated video rendering for this execution.")

        # Render dedicado PathCRF (single panel) con slots, raw edge y acción postprocesada.
        try:
            actions_conf = tracker_conf.get("actions", {}) if isinstance(tracker_conf, dict) else {}
            fps_for_actions = float(actions_conf.get("fps", 25.0))
            detector_for_render = ActionsDetector(ActionsDetectorConfig(fps=fps_for_actions))
            frame_count = len(tracks.get("player", []))
            for fi in range(frame_count):
                detector_for_render.update(
                    fi,
                    {
                        "tracks_frame": {
                            "player": tracks.get("player", [])[fi],
                            "goalkeeper": tracks.get("goalkeeper", [])[fi],
                            "referee": tracks.get("referee", [])[fi],
                            "ball": tracks.get("ball", [])[fi],
                        },
                        "possession": tracks.get("possession", [])[fi] if fi < len(tracks.get("possession", [])) else {},
                    },
                )
            tracking_df, conversion_summary = detector_for_render.build_tracking_dataframe()

            edge_rows = []
            event_rows = []
            for fi, payload in enumerate(tracks.get("actions_incremental", [])):
                if not isinstance(payload, dict):
                    continue
                raw = payload.get("raw_edge")
                if isinstance(raw, dict) and raw.get("edge_src") is not None and raw.get("edge_dst") is not None:
                    edge_rows.append(
                        {
                            "frame_id": int(fi),
                            "edge_src": raw.get("edge_src"),
                            "edge_dst": raw.get("edge_dst"),
                            "edge_team": raw.get("edge_team"),
                        }
                    )
                post = payload.get("confirmed_action")
                if isinstance(post, dict) and post:
                    event_rows.append(
                        {
                            "frame_id": int(fi),
                            "event_type": post.get("event_type"),
                            "player_id": post.get("player_slot_id") or post.get("player_id"),
                            "receiver_id": post.get("receiver_slot_id") or post.get("receiver_id"),
                        }
                    )
            edge_df = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(columns=["frame_id", "edge_src", "edge_dst", "edge_team"])
            events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame(columns=["frame_id", "event_type", "player_id", "receiver_id"])
            output_path_obj = Path(output)
            pathcrf_video_output = output_path_obj.with_name(f"{output_path_obj.stem}_pathcrf.mp4")
            PathCRFDrawer().render_tracking_and_edges(
                tracking=tracking_df,
                edge_sequence=edge_df,
                events=events_df,
                output_path=pathcrf_video_output,
                fps=fps_for_actions,
                video_path=video_path,
                tracks_path=output_path_named,
                conversion_summary={
                    "person_slot_assignments": dict(conversion_summary.person_slot_assignments),
                    "referee_slot_assignments": dict(conversion_summary.referee_slot_assignments),
                },
                show=False,
            )
            logger.info(f"PathCRF dedicated video saved to: {pathcrf_video_output}")
        except Exception:
            logger.exception("No se pudo generar el video dedicado de PathCRF.")

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
        if online_commentary_bridge is not None:
            try:
                online_commentary_bridge.finalize(output_path_named)
            except Exception:
                logger.exception("Fallo en el flush final del bridge PathCRF live.")
        if position_phase is not None and position_phase.enabled:
            role_frame_df, role_player_df, role_greedy_df = (
                position_phase.build_role_export_dataframes(tracks)
            )
            save_dataframe_csv(role_frame_df, role_frame_csv_path, logger, "Frame role predictions CSV")
            copy_output_artifact(role_frame_csv_path, role_frame_csv_path_legacy, logger, "Frame role predictions CSV")
            save_dataframe_csv(role_player_df, role_player_csv_path, logger, "Player role summary CSV")
            copy_output_artifact(role_player_csv_path, role_player_csv_path_legacy, logger, "Player role summary CSV")
            save_dataframe_csv(role_greedy_df, role_greedy_csv_path, logger, "Greedy role diagnostics CSV")
            copy_output_artifact(role_greedy_csv_path, role_greedy_csv_path_legacy, logger, "Greedy role diagnostics CSV")
        save_summary(summary, summary_path, logger)
        if not bool(getattr(args, "skip_metrics_dataset", False)):
            upsert_tracking_metrics_dataset(
                metrics_dataset_path,
                video_path,
                experiment_label,
                output_path_named,
                summary_path,
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
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1

    logger.info("Process completed successfully")
    return 0
