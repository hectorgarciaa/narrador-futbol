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
from football_ai.actions.rolling import RollingActionsConfig, RollingActionsPhase

from .paths import (
    build_output_video_path,
    build_debug_frames_output_path,
    build_tracking_metrics_output_paths,
    build_tracks_output_paths,
    resolve_video_path,
)
from football_ai.commentaries.phase import CommentaryPhase, CommentaryPhaseConfig
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


def _reencode_to_h264(video_path, logger):
    try:
        import subprocess
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_exe = str(get_ffmpeg_exe())
    except ImportError:
        return
    video_path = Path(video_path)
    if not video_path.exists():
        return
    temp_out = video_path.with_suffix(".tmp.mp4")
    try:
        subprocess.run(
            [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                "-profile:v", "main", "-level:v", "4.1",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                "-movflags", "+faststart",
                str(temp_out),
            ],
            check=True,
        )
        temp_out.replace(video_path)
        logger.info(f"Re-encoded to H.264 for browser: {video_path}")
    except Exception:
        pass  # Original file remains


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


def _resolve_position_infering_device(position_phase):
    if position_phase is None:
        return "cpu (position infering disabled)"
    role_session = getattr(position_phase, "role_session", None)
    if role_session is None:
        return "cpu (position infering sin backend torch)"
    runtime_device = getattr(role_session, "device", "cpu")
    return _resolve_runtime_device_label(runtime_device)

def _build_rolling_output_dir(config, video_path) -> Path:
    stem = Path(video_path).stem
    for suffix in ("_tracks", "_tracking", "_edge_sequence"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return (config.project_root / "output" / "actions" / "rolling_online" / stem).resolve()


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
        actions_mode = str(actions_conf.get("mode", "legacy")).strip().lower()
        logger.info("Actions mode: %s", actions_mode)
        if bool(actions_conf.get("enabled", True)):
            if actions_mode != "rolling":
                raise ValueError(f"Actions mode '{actions_mode}' no soportado. Solo se admite 'rolling'.")
            rolling_output_dir = _build_rolling_output_dir(config, video_path)
            actions_phase = RollingActionsPhase(
                RollingActionsConfig(
                    enabled=True,
                    fps=float(actions_conf.get("fps", 25.0)),
                    cadence_frames=max(1, int(actions_conf.get("cadence_frames", 40))),
                    min_frames_warmup=max(1, int(actions_conf.get("min_frames_warmup", 50))),
                    emit_delay_frames=max(0, int(actions_conf.get("emit_delay_frames", 50))),
                    emit_frames=max(1, int(actions_conf.get("emit_frames", 40))),
                    repo_path=Path(str(actions_conf.get("repo_path", "external/pathcrf"))),
                    trial=int(actions_conf.get("trial", 120)),
                    model_file=str(actions_conf.get("model_file", "state_dict_best_acc.pt")),
                    device=str(actions_conf.get("device", "auto")),
                    use_crf=bool(actions_conf.get("use_crf", True)),
                    decode=str(actions_conf.get("decode", "indep")),
                    window_seconds=float(actions_conf.get("window_seconds", 10.0)),
                    sample_freq=int(actions_conf.get("sample_freq", 5)),
                    min_event_duration=int(actions_conf.get("min_event_duration", 10)),
                    smooth_edges=bool(actions_conf.get("smooth_edges", False)),
                    export_debug=bool(actions_conf.get("export_debug", False)),
                    output_dir=str(rolling_output_dir),
                    async_enabled=bool(actions_conf.get("async_enabled", True)),
                    max_workers=max(1, int(actions_conf.get("max_workers", 1))),
                    drop_policy=str(actions_conf.get("drop_policy", "latest")).strip().lower(),
                snapshot_window_frames=(
                    int(actions_conf["snapshot_window_frames"])
                    if actions_conf.get("snapshot_window_frames") is not None
                    else None
                ),
            )
            )

        commentary_phase = None
        commentary_conf = dict(tracker_conf.get("commentary") or {})
        commentary_enabled = bool(commentary_conf.get("enabled", False))
        commentary_enabled = (
            commentary_enabled
            or bool(getattr(args, "commentary", False))
        )
        if getattr(args, "no_commentary", False):
            commentary_enabled = False
        if commentary_enabled:
            from football_ai.commentaries.launcher import ensure_llama_server_running

            llama_base_url = ensure_llama_server_running()
            if llama_base_url:
                logger.info("Commentary LLM backend: %s", llama_base_url)
                if not commentary_conf.get("llm_base_url"):
                    commentary_conf["llm_base_url"] = llama_base_url

            commentary_output_dir = str(rolling_output_dir / "commentary")
            commentary_phase = CommentaryPhase(CommentaryPhaseConfig(
                enabled=True,
                generate_text=bool(commentary_conf.get("generate_text", True)),
                generate_audio=bool(
                    getattr(args, "commentary_audio", None)
                    if hasattr(args, "commentary_audio") and getattr(args, "commentary_audio", None) is not None
                    else commentary_conf.get("generate_audio", True)
                ),
                llm_model=str(getattr(args, "commentary_llm_model", None) or commentary_conf.get("llm_model", "gemma4-q4ks-text")),
                llm_base_url=commentary_conf.get("llm_base_url"),
                llm_temperature=float(commentary_conf.get("llm_temperature", 0.7)),
                tts_backend=str(getattr(args, "commentary_tts_backend", None) or commentary_conf.get("tts_backend", "xtts")),
                speaker_wavs=list(commentary_conf.get("speaker_wavs") or []),
                female_speaker_wavs=list(commentary_conf.get("female_speaker_wavs") or []),
                alternate_voices=bool(commentary_conf.get("alternate_voices", False)),
                elevenlabs_api_key=commentary_conf.get("elevenlabs_api_key"),
                elevenlabs_voice_id=commentary_conf.get("elevenlabs_voice_id"),
                elevenlabs_female_voice_id=commentary_conf.get("elevenlabs_female_voice_id"),
                elevenlabs_model_id=commentary_conf.get("elevenlabs_model_id"),
                elevenlabs_output_format=commentary_conf.get("elevenlabs_output_format"),
                elevenlabs_language_code=commentary_conf.get("elevenlabs_language_code"),
                elevenlabs_stability=commentary_conf.get("elevenlabs_stability"),
                elevenlabs_similarity_boost=commentary_conf.get("elevenlabs_similarity_boost"),
                elevenlabs_style=commentary_conf.get("elevenlabs_style"),
                elevenlabs_speed=commentary_conf.get("elevenlabs_speed"),
                elevenlabs_use_speaker_boost=commentary_conf.get("elevenlabs_use_speaker_boost"),
                elevenlabs_optimize_streaming_latency=commentary_conf.get("elevenlabs_optimize_streaming_latency"),
                fps=float(actions_conf.get("fps", 25.0)),
                output_dir=str(commentary_output_dir),
                max_workers=max(1, int(commentary_conf.get("max_workers", 2))),
                drop_policy=str(commentary_conf.get("drop_policy", "latest")),
                max_queue_size=max(1, int(commentary_conf.get("max_queue_size", 8))),
                skip_event_types=list(commentary_conf.get("skip_event_types", ["out", "unknown"])),
                deduplicate_consecutive=bool(commentary_conf.get("deduplicate_consecutive", True)),
            ))
            logger.info("Commentary phase enabled (mode: %s, audio: %s, tts: %s)",
                        "online", commentary_phase.config.generate_audio, commentary_phase.config.tts_backend)
        else:
            logger.info("Commentary phase disabled.")

        frame_hook = None

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
                f"[runtime] commentary: {'enabled' if commentary_phase else 'disabled'}",
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
            "actions_packets": [],
            "commentary_packets": [],
        }
        visual_debug_frames = []

        tracking_phase.reset()
        posession_phase.reset()
        if position_phase:
            position_phase.reset()
        if actions_phase:
            actions_phase.reset()
        if commentary_phase:
            commentary_phase.reset()

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

            commentary_ms = 0.0
            if commentary_phase:
                commentary_packet, commentary_ms = commentary_phase.process(final_packet)
                final_packet = commentary_packet
            
            tracks_frame = final_packet["clean"]["tracks_frame"]
            for cls in ["player", "goalkeeper", "referee", "ball"]:
                tracks[cls].append(dict(tracks_frame[cls]))
            
            possession_info = final_packet["clean"].get("possession", {})
            tracks["possession"].append(dict(possession_info))
            actions_pkt = final_packet["clean"].get("actions_packet", {})
            tracks["actions_packets"].append(dict(actions_pkt))
            commentary_pkt = final_packet["clean"].get("commentary_packet", {})
            tracks["commentary_packets"].append(dict(commentary_pkt))

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
                total_ms += actions_ms + commentary_ms
                pos_inf = f" | PosInf: {position_ms:.1f}ms" if position_phase else ""
                act_extra = ""
                if actions_phase:
                    actions_pkt = final_packet["clean"].get("actions_packet", {})
                    if actions_pkt.get("checkpoint_completed"):
                        chk_ms = actions_pkt.get("timings", {}).get("total_checkpoint_ms", 0)
                        ev_count = len(actions_pkt.get("emitted_events", []))
                        ed_count = len(actions_pkt.get("emitted_edges", []))
                        le = actions_pkt.get("latest_event") or {}
                        ev_type = le.get("event_type", "?")
                        ev_src = le.get("canonical_src", "?")
                        ev_dst = le.get("canonical_dst", "?")
                        act_extra = f" (chk: {chk_ms:.0f}ms, edges={ed_count}, events={ev_count}, event={ev_type} {ev_src}->{ev_dst})"
                    elif actions_pkt.get("async_pending"):
                        act_extra = " (async pending)"
                act_inf = f" | Actions: {actions_ms:.1f}ms{act_extra}" if actions_phase else ""
                com_inf = ""
                if commentary_phase:
                    com_pkt = final_packet["clean"].get("commentary_packet", {})
                    com_inf = f" | Commentary: {commentary_ms:.1f}ms (q={com_pkt.get('queued',0)} r={com_pkt.get('running',0)} ok={com_pkt.get('completed',0)} fail={com_pkt.get('failed',0)})"
                
                print(
                    f"[frame {frame_index:04d}] "
                    f"Detect: {d_ms:.1f}ms | Proj: {p_ms:.1f}ms | Filter: {f_ms:.1f}ms | "
                    f"Id: {i_ms:.1f}ms | Byte: {b_ms:.1f}ms | Canon: {c_ms:.1f}ms | "
                    f"Poss: {posession_ms:.1f}ms{pos_inf}{act_inf}{com_inf} || Total AI: {total_ms:.1f}ms",
                    flush=True
                )

            # frame_hook removed — commentary now handled by CommentaryPhase

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
                print_equipos=bool(getattr(args, "print_equipos", True)),
            )
            logger.info(f"Video with tracks saved to: {output}")
            _reencode_to_h264(output, logger)
        else:
            logger.info("Skipping annotated video rendering for this execution.")

        # Save rolling artifacts if rolling mode
        rolling_fields: dict[str, Any] = {}
        rolling_artifacts: dict[str, Any] = {}
        if actions_mode == "rolling" and actions_phase is not None:
            try:
                rolling_artifacts = actions_phase.build_result(
                    _build_rolling_output_dir(config, video_path)
                )
                logger.info("Rolling actions completed: %d emitted edges, %d checkpoint%s",
                            len(rolling_artifacts.get("emitted_edges_df", [])),
                            actions_phase._completed_checkpoints,
                            "s" if actions_phase._completed_checkpoints != 1 else "")
                rolling_fields["rolling_mode"] = "rolling"
                rolling_fields["rolling_emitted_edges"] = str(rolling_artifacts.get("emitted_edges", ""))
                rolling_fields["rolling_checkpoints"] = str(rolling_artifacts.get("checkpoints", ""))
            except Exception:
                logger.exception("No se pudieron guardar los artefactos del rolling actions.")

        # Save tracks JSON before PathCRF rendering so the drawer can load it
        save_result(tracks, output_path_named, logger)
        if output_path_legacy != output_path_named:
            save_result(tracks, output_path_legacy, logger)

        # Render dedicado PathCRF (single panel) con edges emitidos y eventos postprocesados
        try:
            actions_conf2 = tracker_conf.get("actions", {}) if isinstance(tracker_conf, dict) else {}
            fps_for_actions = float(actions_conf2.get("fps", 25.0))
            output_path_obj = Path(output)
            pathcrf_video_output = output_path_obj.with_name(f"{output_path_obj.stem}_pathcrf.mp4")

            should_render = True

            tracking_df = None
            if hasattr(actions_phase, '_last_tracking_df') and actions_phase._last_tracking_df is not None:
                tracking_df = actions_phase._last_tracking_df
            else:
                logger.warning("PathCRF render skip: no hay tracking disponible.")
                should_render = False

            if should_render:
                edge_df = rolling_artifacts.get("emitted_edges_df")
                if edge_df is None or (hasattr(edge_df, 'empty') and edge_df.empty):
                    logger.warning("PathCRF render skip: no hay edges emitidos.")
                    should_render = False

            if should_render:
                if "frame_id" not in edge_df.columns:
                    edge_df = edge_df.reset_index()
                if "frame_id" not in edge_df.columns:
                    edge_df["frame_id"] = range(len(edge_df))

                # Build expanded events df: each event occupies [start_frame, end_frame]
                postproc_df = rolling_artifacts.get("postprocessed_actions_df", pd.DataFrame())
                expanded_rows = []
                if not postproc_df.empty:
                    for _, ev in postproc_df.iterrows():
                        sf = int(ev.get("start_frame", ev.get("frame_id", 0)))
                        ef = int(ev.get("end_frame", ev.get("frame_id", 0)))
                        for f in range(sf, ef + 1):
                            expanded_rows.append({**ev.to_dict(), "frame_id": f})
                events_df = pd.DataFrame(expanded_rows) if expanded_rows else postproc_df

                conversion_payload = {}
                if hasattr(actions_phase, '_last_person_slots'):
                    conversion_payload = {
                        "person_slot_assignments": dict(actions_phase._last_person_slots),
                        "referee_slot_assignments": dict(actions_phase._last_referee_slots),
                    }

            if should_render:
                PathCRFDrawer().render_tracking_and_edges(
                    tracking=tracking_df,
                    edge_sequence=edge_df,
                    events=events_df,
                    output_path=pathcrf_video_output,
                    fps=fps_for_actions,
                    video_path=video_path,
                    tracks_path=output_path_named,
                    conversion_summary=conversion_payload,
                    show=False,
                )
                logger.info(f"PathCRF dedicated video saved to: {pathcrf_video_output}")
                _reencode_to_h264(pathcrf_video_output, logger)
        except Exception:
            logger.exception("No se pudo generar el video dedicado de PathCRF.")

        # Evaluation
        logger.info("Evaluating tracks...")
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(["player", "ball"], tracks)
        summary = dict(evaluation["player"]["summary"])
        summary.update(rolling_fields)
        ball_summary = evaluation.get("ball", {}).get("summary", {})
        summary["ball_coverage"] = float(ball_summary.get("mean_coverage", 0.0))
        summary["ball_tracks"] = int(ball_summary.get("num_tracks", 0))
        if role_postprocess_result is not None:
            summary["position_role_online_applied"] = True
            for key, value in role_postprocess_result.items():
                summary[f"special_seed_{key}"] = value

        if commentary_phase is not None:
            try:
                commentary_summary = commentary_phase.summary()
                logger.info("Commentary summary: %s", commentary_summary)
                summary["commentary"] = commentary_summary
            except Exception:
                logger.exception("No se pudo generar el summary de commentary.")
            try:
                commentary_artifacts = commentary_phase.build_result(
                    rolling_output_dir / "commentary"
                )
                logger.info("Commentary artifacts: %s", commentary_artifacts)
            except Exception:
                logger.exception("No se pudieron guardar los artefactos de commentary.")
            try:
                manifest_path = rolling_output_dir / "commentary" / "commentary_manifest.jsonl"
                if manifest_path.exists() and manifest_path.stat().st_size > 0:
                    from football_ai.commentaries.deferred_media import assemble_deferred_commentary_video
                    commentary_audio_path = rolling_output_dir / "commentary" / "commentary_track.wav"
                    commentary_video_out = Path(output).with_name(f"{Path(output).stem}_commentary.mp4")
                    assembly = assemble_deferred_commentary_video(
                        str(manifest_path),
                        str(video_path),
                        str(commentary_audio_path),
                        str(commentary_video_out),
                    )
                    logger.info("Commentary video saved to: %s (events=%d, duration=%.1fs)",
                                assembly.output_video_path, assembly.event_count, assembly.video_duration_seconds)
                    summary["commentary_video"] = str(assembly.output_video_path)
            except Exception:
                logger.exception("No se pudo ensamblar el video con comentarios.")

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
