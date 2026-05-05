#!/usr/bin/env python3
"""
Interfaz web ligera para introducir alineaciones y lanzar `scripts/track.py`.
"""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request
from urllib.parse import parse_qs, quote, urlparse

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import get_config
from football_ai.commentaries.generator import (
    CommentaryGenerator,
    DEFAULT_COMMENTARY_TEMPERATURE,
)
from football_ai.commentaries.voice import (
    DEFAULT_ELEVENLABS_LANGUAGE_CODE,
    DEFAULT_ELEVENLABS_MODEL_ID,
    DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    DEFAULT_TTS_BACKEND,
    TTS_BACKEND_CHOICES,
    CommentaryAudioPipeline,
    build_voice_synthesizer,
)
from football_ai.positions import (
    LineupSpecError,
    get_formation_catalog,
    sanitize_video_stem,
    validate_lineup_payload,
)
from football_ai.pipeline.paths import (
    build_output_video_path,
    resolve_video_path,
)


STATIC_ROOT = Path(__file__).resolve().parent / "static"
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

RUNS_ROOT = PROJECT_ROOT / "output" / "interfaz" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
COMMENTARY_CACHE_ROOT = PROJECT_ROOT / "output" / "interfaz" / "commentary_cache"
COMMENTARY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_COMMENTARY_HOST = "127.0.0.1"
DEFAULT_COMMENTARY_PORT = 8788
OLLAMA_SERVE_LOG_PATH = COMMENTARY_CACHE_ROOT / "ollama_serve.log"
LLAMA_CPP_CONFIG_PATH = PROJECT_ROOT / "external" / "llama.cpp" / "config.yaml"
LLAMA_CPP_SERVER_LOG_PATH = COMMENTARY_CACHE_ROOT / "llama_cpp_server.log"
OLLAMA_STARTUP_TIMEOUT_SECONDS = 90.0
OLLAMA_STARTUP_POLL_SECONDS = 0.5

RUNS = {}
RUNS_LOCK = threading.Lock()
DEFAULT_COMMENTARY_MODE = "live"
COMMENTARY_MODE_CHOICES = {"live", "deferred"}
COMMENTARY_CONFIG = None
APP_RUN_DIR_ENV = "NARRADOR_APP_RUN_DIR"
APP_RUN_ID_ENV = "NARRADOR_APP_RUN_ID"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    if path.stat().st_size == 0:
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                items.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return items


def normalize_commentary_mode(value):
    mode = str(value or DEFAULT_COMMENTARY_MODE).strip().lower()
    if mode not in COMMENTARY_MODE_CHOICES:
        raise LineupSpecError(
            f"`commentary_mode` no soportado: {mode}. "
            f"Opciones: {', '.join(sorted(COMMENTARY_MODE_CHOICES))}."
        )
    return mode


def _resolve_optional_local_path(base_dir, value):
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    else:
        path = path.resolve()
    return path


def load_llama_cpp_launch_config(config_path=None):
    resolved_config_path = Path(config_path or LLAMA_CPP_CONFIG_PATH).expanduser().resolve()
    if not resolved_config_path.exists():
        return None

    with open(resolved_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"El config de llama.cpp debe ser un objeto YAML: {resolved_config_path}"
        )

    runtime_cfg = dict(payload.get("runtime") or {})
    server_cfg = dict(payload.get("server") or {})
    model_cfg = dict(payload.get("model") or {})
    config_dir = resolved_config_path.parent

    executable_path = _resolve_optional_local_path(
        config_dir,
        server_cfg.get("executable") or "./llama-server",
    )
    model_path = _resolve_optional_local_path(
        config_dir,
        model_cfg.get("path"),
    )
    host = str(server_cfg.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(server_cfg.get("port") or 8001)
    alias = str(server_cfg.get("alias") or "gemma4-q4ks-text").strip() or "gemma4-q4ks-text"
    extra_args = server_cfg.get("extra_args") or runtime_cfg.get("extra_args") or []
    if not isinstance(extra_args, list):
        raise ValueError(
            f"`server.extra_args` debe ser una lista en {resolved_config_path}"
        )

    return {
        "config_path": resolved_config_path,
        "config_dir": config_dir,
        "enabled": bool(runtime_cfg.get("enabled", True)),
        "executable_path": executable_path,
        "model_path": model_path,
        "host": host,
        "port": port,
        "base_url": f"http://{host}:{port}",
        "alias": alias,
        "ctx_size": server_cfg.get("ctx_size", 512),
        "n_gpu_layers": server_cfg.get("n_gpu_layers", 999),
        "reasoning": str(server_cfg.get("reasoning") or "off").strip() or "off",
        "no_warmup": bool(server_cfg.get("no_warmup", True)),
        "extra_args": [str(item) for item in extra_args],
    }


def resolve_commentary_runtime_settings(args):
    backend = str(args.commentary_backend or "auto").strip().lower() or "auto"
    llama_cpp_config = load_llama_cpp_launch_config(args.llama_cpp_config)
    if backend == "auto":
        if llama_cpp_config is not None and llama_cpp_config.get("enabled", True):
            backend = "llama_cpp"
        else:
            backend = "ollama"
    if backend not in {"ollama", "llama_cpp"}:
        raise ValueError(f"Backend de comentarios no soportado: {backend}")

    if backend == "llama_cpp":
        if llama_cpp_config is None:
            raise FileNotFoundError(
                "No se encontró el fichero de configuración de llama.cpp. "
                f"Esperado en {Path(args.llama_cpp_config).expanduser().resolve()}"
            )
        model = str(args.commentary_model or llama_cpp_config.get("alias") or "").strip()
        if not model:
            raise ValueError("No se pudo resolver el alias del modelo para llama.cpp.")
        base_url = str(
            args.commentary_base_url
            or llama_cpp_config.get("base_url")
            or "http://127.0.0.1:8001"
        ).strip()
        return {
            "backend": backend,
            "model": model,
            "base_url": base_url,
            "llama_cpp_config": llama_cpp_config,
        }

    return {
        "backend": backend,
        "model": str(args.commentary_model or "gemma4:e2b").strip() or "gemma4:e2b",
        "base_url": str(args.commentary_base_url).strip() if args.commentary_base_url else None,
        "llama_cpp_config": None,
    }


class CommentaryConfig:
    """Config holder para generacion de comentarios (intro + pipeline)."""

    def __init__(
        self,
        *,
        backend="auto",
        model="gemma4-q4ks-text",
        temperature=DEFAULT_COMMENTARY_TEMPERATURE,
        base_url=None,
        tts_backend=DEFAULT_TTS_BACKEND,
        elevenlabs_api_key=None,
        elevenlabs_voice_id=None,
        elevenlabs_female_voice_id=None,
        elevenlabs_model_id=None,
        elevenlabs_output_format=None,
        elevenlabs_language_code=None,
        elevenlabs_stability=None,
        elevenlabs_similarity_boost=None,
        elevenlabs_style=None,
        elevenlabs_speed=None,
        elevenlabs_use_speaker_boost=None,
        elevenlabs_optimize_streaming_latency=None,
    ):
        self.backend = str(backend).strip().lower() or "auto"
        self.llama_cpp_config = None
        self.model = str(model).strip() or "gemma4-q4ks-text"
        self.temperature = float(temperature)
        self.base_url = str(base_url).strip() if base_url else None
        self.tts_backend = str(tts_backend or DEFAULT_TTS_BACKEND).strip().lower()
        self.elevenlabs_api_key = str(elevenlabs_api_key).strip() if elevenlabs_api_key else None
        self.elevenlabs_voice_id = str(elevenlabs_voice_id).strip() if elevenlabs_voice_id else None
        self.elevenlabs_female_voice_id = str(elevenlabs_female_voice_id).strip() if elevenlabs_female_voice_id else None
        self.elevenlabs_model_id = str(elevenlabs_model_id).strip() if elevenlabs_model_id else None
        self.elevenlabs_output_format = str(elevenlabs_output_format).strip() if elevenlabs_output_format else None
        self.elevenlabs_language_code = str(elevenlabs_language_code).strip() if elevenlabs_language_code else None
        self.elevenlabs_stability = elevenlabs_stability
        self.elevenlabs_similarity_boost = elevenlabs_similarity_boost
        self.elevenlabs_style = elevenlabs_style
        self.elevenlabs_speed = elevenlabs_speed
        self.elevenlabs_use_speaker_boost = elevenlabs_use_speaker_boost
        self.elevenlabs_optimize_streaming_latency = elevenlabs_optimize_streaming_latency

    @property
    def backend_display_name(self):
        if self.backend == "llama_cpp":
            return "llama.cpp"
        return self.backend

    def build_commentary_generator(self):
        return CommentaryGenerator(
            model=self.model,
            temperature=self.temperature,
            base_url=self._llm_base_url(),
        )

    def build_voice_synthesizer(self, *, alternate_voices=True):
        return build_voice_synthesizer(
            tts_backend=self.tts_backend,
            alternate_voices=bool(alternate_voices),
            elevenlabs_api_key=self.elevenlabs_api_key,
            elevenlabs_voice_id=self.elevenlabs_voice_id,
            elevenlabs_female_voice_id=self.elevenlabs_female_voice_id,
            elevenlabs_model_id=self.elevenlabs_model_id,
            elevenlabs_output_format=self.elevenlabs_output_format,
            elevenlabs_language_code=self.elevenlabs_language_code,
            elevenlabs_stability=self.elevenlabs_stability,
            elevenlabs_similarity_boost=self.elevenlabs_similarity_boost,
            elevenlabs_style=self.elevenlabs_style,
            elevenlabs_speed=self.elevenlabs_speed,
            elevenlabs_use_speaker_boost=self.elevenlabs_use_speaker_boost,
            elevenlabs_optimize_streaming_latency=self.elevenlabs_optimize_streaming_latency,
        )

    def _llm_base_url(self):
        if self.base_url:
            return self.base_url.rstrip("/")
        env_base_url = str(os.environ.get("LLAMA_CPP_BASE_URL") or "").strip()
        if not env_base_url:
            return "http://127.0.0.1:8001"
        if re.match(r"^https?://", env_base_url, flags=re.IGNORECASE):
            return env_base_url.rstrip("/")
        return f"http://{env_base_url}".rstrip("/")


def commentary_config_from_args(args):
    commentary_runtime = resolve_commentary_runtime_settings(args)
    config = CommentaryConfig(
        backend=commentary_runtime["backend"],
        model=commentary_runtime["model"],
        temperature=args.commentary_temperature,
        base_url=commentary_runtime["base_url"],
        tts_backend=args.commentary_tts_backend,
        elevenlabs_api_key=args.elevenlabs_api_key,
        elevenlabs_voice_id=args.elevenlabs_voice_id,
        elevenlabs_female_voice_id=args.elevenlabs_female_voice_id,
        elevenlabs_model_id=args.elevenlabs_model_id,
        elevenlabs_output_format=args.elevenlabs_output_format,
        elevenlabs_language_code=args.elevenlabs_language_code,
        elevenlabs_stability=args.elevenlabs_stability,
        elevenlabs_similarity_boost=args.elevenlabs_similarity_boost,
        elevenlabs_style=args.elevenlabs_style,
        elevenlabs_speed=args.elevenlabs_speed,
        elevenlabs_use_speaker_boost=args.elevenlabs_use_speaker_boost,
        elevenlabs_optimize_streaming_latency=args.elevenlabs_optimize_streaming_latency,
    )
    config.llama_cpp_config = commentary_runtime.get("llama_cpp_config")
    return config


def build_intro_event(team_name=None, opponent_team_name=None):
    event = {
        "action": "intro",
        "event_time_s": 0.0,
        "player_name": "",
        "player_position": "",
    }
    team_name = str(team_name or "").strip()
    opponent_team_name = str(opponent_team_name or "").strip()
    if team_name:
        event["team_name"] = team_name
    if opponent_team_name:
        event["opponent_team_name"] = opponent_team_name
    return event


def build_match_intro_event(lineup_spec):
    teams = list((lineup_spec or {}).get("teams") or [])
    team_names = [
        str(team.get("team_name") or "").strip()
        for team in teams[:2]
        if str(team.get("team_name") or "").strip()
    ]
    if len(team_names) >= 2:
        return build_intro_event(
            team_name=team_names[0],
            opponent_team_name=team_names[1],
        )
    if team_names:
        return build_intro_event(team_name=team_names[0])
    return build_intro_event()


def _looks_like_bad_intro_commentary(commentary):
    text = str(commentary or "").strip()
    if not text:
        return True
    normalized = text.casefold()
    suspicious_patterns = (
        r"\bjugador\b",
        r"\bposicion\b",
        r"\bposición\b",
        r"\brol\b",
        r"\b_\b",
    )
    if any(re.search(pattern, normalized) for pattern in suspicious_patterns):
        return True
    return "_" in text


def _fallback_intro_text(event=None):
    event = dict(event or {})
    team_name = str(event.get("team_name") or "").strip()
    opponent_team_name = str(event.get("opponent_team_name") or "").strip()
    if team_name and opponent_team_name:
        return (
            f"Bienvenidos, ya esta todo preparado para disfrutar del partido entre "
            f"{team_name} y {opponent_team_name}."
        )
    if team_name:
        return (
            f"Bienvenidos, ya esta todo preparado para disfrutar de un partido con "
            f"{team_name} como protagonista."
        )
    return (
        "Bienvenidos, ya esta todo preparado para disfrutar de un partido que "
        "promete emociones fuertes."
    )


def generate_intro_commentary_locally(event, audio_path, commentary_cfg):
    event = dict(event or {})
    audio_path = Path(audio_path)
    generator = commentary_cfg.build_commentary_generator()
    voice_synthesizer = commentary_cfg.build_voice_synthesizer(
        alternate_voices=True,
    )
    voice_synthesizer.prepare()
    pipeline = CommentaryAudioPipeline(
        commentary_generator=generator,
        voice_synthesizer=voice_synthesizer,
    )

    try:
        result = pipeline.generate_to_file(
            event,
            audio_path=audio_path,
        )
        if _looks_like_bad_intro_commentary(result.commentary):
            raise RuntimeError(
                "El modelo devolvio un intro con placeholders o plantilla legacy."
            )
        return {
            "commentary": result.commentary,
            "audio_path": str(result.audio_path),
            "model": result.commentary_result.model,
            "tts_model": result.tts_model,
            "voice_label": result.voice_label,
            "llm_seconds": round(result.llm_seconds or 0.0, 3),
            "tts_seconds": round(result.tts_seconds or 0.0, 3),
            "total_seconds": round(result.total_seconds or 0.0, 3),
        }
    except Exception:
        fallback_text = _fallback_intro_text(event)
        tts_start = time.perf_counter()
        audio_path = voice_synthesizer.synthesize_to_file(
            fallback_text,
            audio_path,
        )
        tts_seconds = time.perf_counter() - tts_start
        return {
            "commentary": fallback_text,
            "audio_path": str(audio_path),
            "model": f"{commentary_cfg.model}:intro-fallback",
            "tts_model": getattr(voice_synthesizer, "model_name", None),
            "voice_label": getattr(voice_synthesizer, "last_voice_label", None),
            "llm_seconds": 0.0,
            "tts_seconds": round(tts_seconds, 3),
            "total_seconds": round(tts_seconds, 3),
        }


def generate_intro_commentary_payload(event, audio_path, *, timeout=300.0, commentary_cfg=None):
    if commentary_cfg is None:
        raise RuntimeError("CommentaryConfig no configurado.")
    audio_path = Path(audio_path)
    try:
        return generate_intro_commentary_locally(
            event,
            audio_path,
            commentary_cfg,
        )
    except Exception:
        raise RuntimeError("No se pudo generar el intro inicial.")


def build_run_paths(run_id):
    run_dir = RUNS_ROOT / str(run_id)
    return {
        "run_dir": run_dir,
        "status_path": run_dir / "status.json",
        "spec_path": run_dir / "lineup_spec.json",
        "log_path": run_dir / "track.log",
        "commentary_dir": run_dir / "commentaries",
        "commentary_audio_dir": run_dir / "commentaries" / "audio",
        "commentary_manifest_path": run_dir / "commentaries" / "events_manifest.jsonl",
        "commentary_intro_audio_path": run_dir / "commentaries" / "audio" / "000_intro.wav",
        "commentary_intro_meta_path": run_dir / "commentaries" / "intro.json",
        "commentary_track_audio_path": run_dir / "commentaries" / "commentary_track.wav",
    }


def slugify_run_part(value, *, fallback="run", max_length=40):
    slug = sanitize_video_stem(str(value or "").strip()).replace("_", "-").lower()
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = fallback
    return slug[:max_length].strip("-") or fallback


def build_interface_run_id(lineup_spec):
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    teams = list(lineup_spec.get("teams") or [])
    team_names = [
        slugify_run_part(
            team.get("team_name"),
            fallback=f"equipo{idx + 1}",
            max_length=24,
        )
        for idx, team in enumerate(teams[:2])
    ]
    if len(team_names) >= 2:
        matchup = f"{team_names[0]}-vs-{team_names[1]}"
    elif team_names:
        matchup = team_names[0]
    else:
        matchup = "sin-equipos"
    video = slugify_run_part(
        lineup_spec.get("video_source"),
        fallback="video",
        max_length=36,
    )

    for _ in range(20):
        suffix = uuid.uuid4().hex[:4]
        run_id = f"{timestamp}_{matchup}_{video}_{suffix}"
        if not build_run_paths(run_id)["run_dir"].exists():
            return run_id
    return f"{timestamp}_{matchup}_{video}_{uuid.uuid4().hex[:8]}"


def resolve_tracking_output_video_path(video_source):
    config = get_config()
    resolved_video_path, _ = resolve_video_path(config, video_source)
    return Path(build_output_video_path(config, resolved_video_path)).resolve()


def resolve_pathcrf_output_video_path(video_source):
    tracking_path = resolve_tracking_output_video_path(video_source)
    return tracking_path.with_name(f"{tracking_path.stem}_pathcrf.mp4")


def resolve_commentary_output_video_path(video_source):
    tracking_path = resolve_tracking_output_video_path(video_source)
    return tracking_path.with_name(f"{tracking_path.stem}_commentary.mp4")


def build_artifact_url(run_id, artifact_path):
    run_paths = build_run_paths(run_id)
    resolved_path = Path(artifact_path).expanduser().resolve()
    run_root = run_paths["run_dir"].resolve()
    relative_path = resolved_path.relative_to(run_root)
    return f"/api/runs/{run_id}/artifacts/{quote(relative_path.as_posix(), safe='/')}"


def serialize_commentary_event_for_client(run_id, index, entry):
    item = dict(entry)
    item["index"] = int(index)
    audio_path = item.get("audio_path")
    if audio_path:
        try:
            item["audio_url"] = build_artifact_url(run_id, audio_path)
        except Exception:
            item["audio_url"] = None
    return item


def config_video_shortcuts():
    config = get_config()
    data_paths = config.paths.get("data", {})
    shortcuts = []
    for key in sorted(data_paths.keys()):
        if data_paths.get(key) is None:
            continue
        resolved_path = config.get_path("paths", "data", key)
        if resolved_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
            continue
        shortcuts.append(
            {
                "key": str(key),
                "path": str(resolved_path),
            }
        )
    return shortcuts


def commentary_config():
    return COMMENTARY_CONFIG


def initial_commentary_payload(run_id, run_paths, commentary_mode):
    return {
        "mode": commentary_mode,
        "manifest_path": str(run_paths["commentary_manifest_path"]),
        "events_api_path": f"/api/runs/{run_id}/commentary-events",
        "audio_dir": str(run_paths["commentary_audio_dir"]),
        "commentary_track_path": str(run_paths["commentary_track_audio_path"]),
        "intro_status": "pending",
        "deferred_mux_status": "pending",
    }


def _resolve_video_path_from_candidates(*candidates):
    for candidate in candidates:
        if not candidate:
            continue
        resolved_path = Path(candidate).expanduser().resolve()
        if resolved_path.exists() and resolved_path.is_file():
            return resolved_path
    return None


def resolve_all_result_video_paths(status):
    if not isinstance(status, dict):
        return None, None, None
    commentary = dict(status.get("commentary") or {})
    commentary_mode = str(commentary.get("mode") or "").strip().lower()
    deferred_status = str(commentary.get("deferred_mux_status") or "").strip().lower()
    is_done = str(status.get("status") or "").strip().lower() not in {"queued", "running"}

    tracking = None
    pathcrf = _resolve_video_path_from_candidates(status.get("pathcrf_video_path"))
    commentary_video = None

    if commentary_mode == "deferred":
        if deferred_status == "ready":
            tracking = _resolve_video_path_from_candidates(commentary.get("deferred_video_path"))
        elif deferred_status == "failed":
            tracking = _resolve_video_path_from_candidates(status.get("output_video_path"))
    elif is_done:
        tracking = _resolve_video_path_from_candidates(status.get("output_video_path"))

    if is_done:
        commentary_video = _resolve_video_path_from_candidates(status.get("commentary_video_path"))

    return tracking, pathcrf, commentary_video


def attach_intro_to_run(run_id, lineup_spec, commentary_mode):
    run_paths = build_run_paths(run_id)
    run_paths["commentary_audio_dir"].mkdir(parents=True, exist_ok=True)

    intro_event = build_match_intro_event(lineup_spec)
    has_team_intro = bool(
        str(intro_event.get("team_name") or "").strip()
        and str(intro_event.get("opponent_team_name") or "").strip()
    )
    if not has_team_intro:
        payload = {
            **initial_commentary_payload(run_id, run_paths, commentary_mode),
            "intro_status": "pending",
            "intro_error": "La intro necesita los nombres de los dos equipos.",
        }
        return payload

    intro_audio_path = run_paths["commentary_intro_audio_path"]
    intro_payload = generate_intro_commentary_payload(
        intro_event,
        intro_audio_path,
        timeout=300.0,
        commentary_cfg=COMMENTARY_CONFIG,
    )
    generated_audio_path = Path(intro_payload["audio_path"]).expanduser().resolve()
    if generated_audio_path.suffix.lower() != intro_audio_path.suffix.lower():
        intro_audio_path = intro_audio_path.with_suffix(generated_audio_path.suffix)
        if generated_audio_path != intro_audio_path:
            shutil.copy2(generated_audio_path, intro_audio_path)
            intro_payload["audio_path"] = str(intro_audio_path)

    write_json(run_paths["commentary_intro_meta_path"], intro_payload)
    append_jsonl(
        run_paths["commentary_manifest_path"],
        {
            "generated_at_utc": intro_payload.get("generated_at_utc") or now_iso(),
            "event": intro_payload.get("event") or intro_event,
            "mode": commentary_mode,
            "metadata": {
                "run_id": str(run_id),
                "source": "interfaz",
                "event_kind": "intro",
                "team_specific_intro": True,
            },
            "commentary": intro_payload.get("commentary"),
            "audio_path": intro_payload.get("audio_path"),
            "model": intro_payload.get("model"),
            "tts_model": intro_payload.get("tts_model"),
            "voice_label": intro_payload.get("voice_label"),
            "text_only": False,
            "llm_seconds": intro_payload.get("llm_seconds"),
            "tts_seconds": intro_payload.get("tts_seconds"),
            "total_seconds": intro_payload.get("total_seconds"),
            "audio_duration_seconds": intro_payload.get("audio_duration_seconds"),
        },
    )

    intro_audio_url = None
    if intro_payload.get("audio_path"):
        intro_audio_url = build_artifact_url(run_id, intro_payload["audio_path"])

    return {
        **initial_commentary_payload(run_id, run_paths, commentary_mode),
        "intro_status": "ready",
        "intro_commentary": intro_payload.get("commentary"),
        "intro_audio_path": intro_payload.get("audio_path"),
        "intro_audio_url": intro_audio_url,
        "intro_event_index": 0,
        "intro_generated_at_utc": intro_payload.get("generated_at_utc"),
        "intro_team_name": intro_event.get("team_name"),
        "intro_opponent_team_name": intro_event.get("opponent_team_name"),
        "llm_seconds": intro_payload.get("llm_seconds"),
        "tts_seconds": intro_payload.get("tts_seconds"),
        "total_seconds": intro_payload.get("total_seconds"),
    }


def initial_status_payload(run_id, lineup_spec, run_paths, commentary_mode):
    teams = lineup_spec["teams"]
    video_source = str(lineup_spec.get("video_source") or "").strip()
    video_stem = sanitize_video_stem(video_source) if video_source else None
    output_video_path = None
    pathcrf_video_path = None
    commentary_video_path = None
    if video_source:
        try:
            output_video_path = str(resolve_tracking_output_video_path(video_source))
            pathcrf_video_path = str(resolve_pathcrf_output_video_path(video_source))
            commentary_video_path = str(resolve_commentary_output_video_path(video_source))
        except Exception:
            output_video_path = None
    return {
        "run_id": str(run_id),
        "status": "queued",
        "created_at_utc": now_iso(),
        "updated_at_utc": now_iso(),
        "video_source": video_source,
        "video_stem_hint": video_stem,
        "output_video_path": output_video_path,
        "pathcrf_video_path": pathcrf_video_path,
        "commentary_video_path": commentary_video_path,
        "teams": [
            {
                "team_name": str(team["team_name"]),
                "team_color": str(team["team_color"]),
                "formation": str(team["formation"]),
            }
            for team in teams
        ],
        "spec_path": str(run_paths["spec_path"]),
        "log_path": str(run_paths["log_path"]),
        "run_dir": str(run_paths["run_dir"]),
        "commentary": initial_commentary_payload(run_id, run_paths, commentary_mode),
    }


def update_status_file(run_id, **changes):
    run_paths = build_run_paths(run_id)
    status = read_json(run_paths["status_path"], default={}) or {}
    status.update(changes)
    status["updated_at_utc"] = now_iso()
    write_json(run_paths["status_path"], status)
    with RUNS_LOCK:
        RUNS[str(run_id)] = status
    return status


def update_commentary_payload(run_id, **changes):
    run_paths = build_run_paths(run_id)
    status = read_json(run_paths["status_path"], default={}) or {}
    commentary = dict(status.get("commentary") or {})
    commentary.update(changes)
    return update_status_file(run_id, commentary=commentary)


def tail_log(log_path, max_lines=80):
    path = Path(log_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def launch_tracking_process(run_id, lineup_spec, commentary_mode=DEFAULT_COMMENTARY_MODE):
    run_paths = build_run_paths(run_id)
    run_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["commentary_audio_dir"].mkdir(parents=True, exist_ok=True)
    write_json(run_paths["spec_path"], lineup_spec)
    status = initial_status_payload(run_id, lineup_spec, run_paths, commentary_mode)
    write_json(run_paths["status_path"], status)

    video_source = str(lineup_spec.get("video_source") or "").strip()
    if not video_source:
        raise LineupSpecError(
            "La interfaz necesita `video_source` para lanzar el tracking."
        )
    output_video_path = resolve_tracking_output_video_path(video_source)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "track.py"),
        video_source,
        "--lineup-spec",
        str(run_paths["spec_path"]),
        "--commentary",
        "--commentary-audio",
        "--profile-phases",
    ]
    process_env = dict(os.environ)
    process_env.update(
        {
            APP_RUN_DIR_ENV: str(run_paths["run_dir"]),
            APP_RUN_ID_ENV: str(run_id),
        }
    )

    log_file = open(run_paths["log_path"], "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=process_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with RUNS_LOCK:
        RUNS[str(run_id)] = {
            **status,
            "status": "running",
            "pid": int(process.pid),
            "command": cmd,
        }
    update_status_file(
        run_id,
        status="running",
        pid=int(process.pid),
        command=cmd,
        started_at_utc=now_iso(),
    )

    def _watch():
        return_code = None
        error_message = None
        try:
            return_code = process.wait()
        except Exception as exc:
            error_message = str(exc)
        finally:
            log_file.close()

        if error_message is not None:
            update_status_file(
                run_id,
                status="failed",
                finished_at_utc=now_iso(),
                return_code=return_code,
                error=error_message,
            )
            return

        final_status = "completed" if int(return_code or 0) == 0 else "failed"
        update_status_file(
            run_id,
            status=final_status,
            finished_at_utc=now_iso(),
            return_code=int(return_code or 0),
        )
        if final_status != "completed":
            return

        # Commentary video is already assembled by the pipeline (from tracking video + audio).
        # Just update the status to reflect completion.
        try:
            commentary_video_path = str(resolve_commentary_output_video_path(video_source))
            if Path(commentary_video_path).exists():
                update_commentary_payload(
                    run_id,
                    deferred_mux_status="ready",
                    deferred_video_path=commentary_video_path,
                )
        except Exception as exc:
            update_commentary_payload(
                run_id,
                deferred_mux_status="failed",
                deferred_mux_error=str(exc),
            )

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    return read_json(run_paths["status_path"])


class InterfaceRequestHandler(BaseHTTPRequestHandler):
    server_version = "NarradorFutbolInterface/0.1"

    def log_message(self, format, *args):  # noqa: A003
        return

    def _send_json(self, payload, status=HTTPStatus.OK, include_body=True):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def _send_file(self, file_path, include_body=True):
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            self._send_json(
                {"error": "Not found"},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"
        total_size = int(file_path.stat().st_size)
        start = 0
        end = max(0, total_size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)$", str(range_header).strip())
            if match:
                start_text, end_text = match.groups()
                if start_text:
                    start = int(start_text)
                if end_text:
                    end = int(end_text)
                elif total_size > 0:
                    end = total_size - 1
                if start >= total_size or start < 0 or end < start:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{total_size}")
                    self.end_headers()
                    return
                end = min(end, total_size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
        content_length = max(0, end - start + 1)

        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
        self.end_headers()
        if include_body:
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _handle_get_run_status(self, run_id, include_body=True):
        run_paths = build_run_paths(run_id)
        status = read_json(run_paths["status_path"], default=None)
        if status is None:
            self._send_json(
                {"error": f"No existe la ejecución {run_id}."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        status["log_tail"] = tail_log(run_paths["log_path"])
        tracking_path, pathcrf_path, commentary_path = resolve_all_result_video_paths(status)
        status["result_video_path"] = str(tracking_path) if tracking_path is not None else None
        status["result_video_url"] = (
            f"/api/runs/{run_id}/result-video"
            if tracking_path is not None
            else None
        )
        status["pathcrf_video_url"] = (
            f"/api/runs/{run_id}/pathcrf-video"
            if pathcrf_path is not None
            else None
        )
        status["commentary_video_url"] = (
            f"/api/runs/{run_id}/commentary-video"
            if commentary_path is not None
            else None
        )
        self._send_json(status, include_body=include_body)

    def _handle_get_commentary_service(self, include_body=True):
        payload = {
            "status": "ready" if COMMENTARY_CONFIG is not None else "unavailable",
            "backend": (COMMENTARY_CONFIG.backend if COMMENTARY_CONFIG else None),
            "model": (COMMENTARY_CONFIG.model if COMMENTARY_CONFIG else None),
            "tts_backend": (COMMENTARY_CONFIG.tts_backend if COMMENTARY_CONFIG else None),
            "intro_policy": "team_names_required",
        }
        self._send_json(payload, include_body=include_body)

    def _handle_get_run_artifact(self, run_id, relative_path, include_body=True):
        run_paths = build_run_paths(run_id)
        run_root = run_paths["run_dir"].resolve()
        artifact_path = (run_root / relative_path).resolve()
        if run_root not in artifact_path.parents and artifact_path != run_root:
            self._send_json(
                {"error": "Ruta de artefacto no permitida."},
                status=HTTPStatus.FORBIDDEN,
                include_body=include_body,
            )
            return
        self._send_file(artifact_path, include_body=include_body)

    def _send_video_file(self, video_path_candidate, include_body=True):
        if not video_path_candidate:
            self._send_json(
                {"error": "El vídeo todavía no está disponible."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        resolved = Path(video_path_candidate).expanduser().resolve()
        project_root = PROJECT_ROOT.resolve()
        if project_root not in resolved.parents and resolved != project_root:
            self._send_json(
                {"error": "Ruta de vídeo no permitida."},
                status=HTTPStatus.FORBIDDEN,
                include_body=include_body,
            )
            return
        self._send_file(resolved, include_body=include_body)

    def _handle_get_run_result_video(self, run_id, include_body=True):
        run_paths = build_run_paths(run_id)
        status = read_json(run_paths["status_path"], default=None)
        if status is None:
            self._send_json(
                {"error": f"No existe la ejecución {run_id}."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        tracking_path, _, _ = resolve_all_result_video_paths(status)
        self._send_video_file(tracking_path, include_body=include_body)

    def _handle_get_run_pathcrf_video(self, run_id, include_body=True):
        run_paths = build_run_paths(run_id)
        status = read_json(run_paths["status_path"], default=None)
        if status is None:
            self._send_json(
                {"error": f"No existe la ejecución {run_id}."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        _, pathcrf_path, _ = resolve_all_result_video_paths(status)
        self._send_video_file(pathcrf_path, include_body=include_body)

    def _handle_get_run_commentary_video(self, run_id, include_body=True):
        run_paths = build_run_paths(run_id)
        status = read_json(run_paths["status_path"], default=None)
        if status is None:
            self._send_json(
                {"error": f"No existe la ejecución {run_id}."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        _, _, commentary_path = resolve_all_result_video_paths(status)
        self._send_video_file(commentary_path, include_body=include_body)

    def _handle_get_commentary_events(self, run_id, query, include_body=True):
        run_paths = build_run_paths(run_id)
        after_raw = (parse_qs(query or "").get("after") or ["-1"])[0]
        try:
            after_index = int(after_raw)
        except ValueError:
            after_index = -1

        events = read_jsonl(run_paths["commentary_manifest_path"])
        selected = [
            serialize_commentary_event_for_client(run_id, idx, entry)
            for idx, entry in enumerate(events)
            if idx > after_index
        ]
        next_after = after_index
        if selected:
            next_after = int(selected[-1]["index"])
        self._send_json(
            {
                "events": selected,
                "next_after": next_after,
                "count": len(selected),
                "manifest_path": str(run_paths["commentary_manifest_path"]),
            },
            include_body=include_body,
        )

    def _handle_request(self, include_body=True):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/commentary-service":
            self._handle_get_commentary_service(include_body=include_body)
            return

        if path == "/api/formations":
            self._send_json(
                {"formations": get_formation_catalog()},
                include_body=include_body,
            )
            return

        if path == "/api/videos":
            self._send_json(
                {"videos": config_video_shortcuts()},
                include_body=include_body,
            )
            return

        if path == "/api/runs":
            run_items = []
            for run_dir in sorted(RUNS_ROOT.glob("*"), reverse=True):
                status = read_json(run_dir / "status.json", default=None)
                if status is not None:
                    run_items.append(status)
            self._send_json({"runs": run_items[:20]}, include_body=include_body)
            return

        if path.startswith("/api/runs/"):
            if path.endswith("/result-video"):
                run_id = path.split("/api/runs/", 1)[1].rsplit("/result-video", 1)[0].strip("/")
                if not run_id:
                    self._send_json(
                        {"error": "Run id inválido."},
                        status=HTTPStatus.BAD_REQUEST,
                        include_body=include_body,
                    )
                    return
                self._handle_get_run_result_video(run_id, include_body=include_body)
                return

            if path.endswith("/pathcrf-video"):
                run_id = path.rsplit("/pathcrf-video", 1)[0].split("/api/runs/", 1)[1].strip("/")
                if not run_id:
                    self._send_json(
                        {"error": "Run id inválido."},
                        status=HTTPStatus.BAD_REQUEST,
                        include_body=include_body,
                    )
                    return
                self._handle_get_run_pathcrf_video(run_id, include_body=include_body)
                return

            if path.endswith("/commentary-video"):
                run_id = path.rsplit("/commentary-video", 1)[0].split("/api/runs/", 1)[1].strip("/")
                if not run_id:
                    self._send_json(
                        {"error": "Run id inválido."},
                        status=HTTPStatus.BAD_REQUEST,
                        include_body=include_body,
                    )
                    return
                self._handle_get_run_commentary_video(run_id, include_body=include_body)
                return

            if "/artifacts/" in path:
                prefix, relative_path = path.split("/artifacts/", 1)
                run_id = prefix.split("/api/runs/", 1)[1].strip("/")
                if not run_id or not relative_path:
                    self._send_json(
                        {"error": "Ruta de artefacto invalida."},
                        status=HTTPStatus.BAD_REQUEST,
                        include_body=include_body,
                    )
                    return
                self._handle_get_run_artifact(run_id, relative_path, include_body=include_body)
                return

            if path.endswith("/commentary-events"):
                run_id = path.split("/api/runs/", 1)[1].rsplit("/commentary-events", 1)[0].strip("/")
                if not run_id:
                    self._send_json(
                        {"error": "Run id inválido."},
                        status=HTTPStatus.BAD_REQUEST,
                        include_body=include_body,
                    )
                    return
                self._handle_get_commentary_events(run_id, parsed.query, include_body=include_body)
                return

            run_id = path.split("/api/runs/", 1)[1].strip("/")
            if not run_id:
                self._send_json(
                    {"error": "Run id inválido."},
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return
            self._handle_get_run_status(run_id, include_body=include_body)
            return

        if path in {"/", ""}:
            self._send_file(STATIC_ROOT / "index.html", include_body=include_body)
            return

        static_candidate = (STATIC_ROOT / path.lstrip("/")).resolve()
        if STATIC_ROOT.resolve() not in static_candidate.parents and static_candidate != STATIC_ROOT.resolve():
            self._send_json(
                {"error": "Ruta no permitida."},
                status=HTTPStatus.FORBIDDEN,
                include_body=include_body,
            )
            return
        self._send_file(static_candidate, include_body=include_body)

    def do_GET(self):  # noqa: N802
        self._handle_request(include_body=True)

    def do_HEAD(self):  # noqa: N802
        self._handle_request(include_body=False)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/runs":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
            commentary_mode = normalize_commentary_mode(payload.get("commentary_mode"))
            lineup_spec = validate_lineup_payload(payload)
            if not str(lineup_spec.get("video_source") or "").strip():
                raise LineupSpecError(
                    "Debes seleccionar un vídeo (`video_source`) antes de ejecutar."
                )
            run_id = build_interface_run_id(lineup_spec)
            status = launch_tracking_process(
                run_id,
                lineup_spec,
                commentary_mode=commentary_mode,
            )
            try:
                commentary_payload = attach_intro_to_run(
                    run_id,
                    lineup_spec,
                    commentary_mode,
                )
            except Exception as exc:
                commentary_payload = {
                    **initial_commentary_payload(
                        run_id,
                        build_run_paths(run_id),
                        commentary_mode,
                    ),
                    "intro_status": "failed",
                    "error": str(exc),
                }
            status = update_status_file(run_id, commentary=commentary_payload)
            self._send_json(status, status=HTTPStatus.CREATED)
        except LineupSpecError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except json.JSONDecodeError:
            self._send_json(
                {"error": "JSON inválido en la petición."},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:  # pragma: no cover - defensivo
            self._send_json(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interfaz web ligera para introducir alineaciones y lanzar tracking."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host a escuchar.")
    parser.add_argument("--port", type=int, default=8767, help="Puerto HTTP.")
    parser.add_argument(
        "--commentary-backend",
        choices=("auto", "ollama", "llama_cpp"),
        default="auto",
        help=(
            "Backend LLM para el servidor de comentarios. "
            "`auto` usa `llama.cpp` si existe `llama.cpp/config.yaml`; "
            "si no, cae a `ollama`."
        ),
    )
    parser.add_argument(
        "--commentary-host",
        default=DEFAULT_COMMENTARY_HOST,
        help="Host para el servidor de comentarios lanzado junto a la interfaz.",
    )
    parser.add_argument(
        "--commentary-port",
        type=int,
        default=DEFAULT_COMMENTARY_PORT,
        help="Puerto para el servidor de comentarios.",
    )
    parser.add_argument(
        "--commentary-model",
        default=None,
        help="Modelo o alias del backend de comentarios.",
    )
    parser.add_argument(
        "--commentary-temperature",
        type=float,
        default=DEFAULT_COMMENTARY_TEMPERATURE,
        help="Temperatura para el LLM de comentarios.",
    )
    parser.add_argument(
        "--commentary-base-url",
        default=None,
        help="URL base del backend LLM para el servidor de comentarios.",
    )
    parser.add_argument(
        "--commentary-tts-backend",
        choices=TTS_BACKEND_CHOICES,
        default=os.environ.get("NARRADOR_COMMENTARY_TTS_BACKEND", DEFAULT_TTS_BACKEND),
        help=(
            "Backend de voz para el servidor de comentarios. "
            "`xtts` mantiene el flujo local; `elevenlabs` usa la API streaming."
        ),
    )
    parser.add_argument(
        "--elevenlabs-api-key",
        default=None,
        help="API key de ElevenLabs. Si no se indica, usa ELEVENLABS_API_KEY.",
    )
    parser.add_argument(
        "--elevenlabs-voice-id",
        default=None,
        help="ID de la voz de ElevenLabs. Si no se indica, usa ELEVENLABS_VOICE_ID.",
    )
    parser.add_argument(
        "--elevenlabs-female-voice-id",
        default=None,
        help=(
            "ID de voz femenina de ElevenLabs para alternar comentaristas. "
            "Si no se indica, usa ELEVENLABS_FEMALE_VOICE_ID."
        ),
    )
    parser.add_argument(
        "--elevenlabs-model-id",
        default=None,
        help=(
            "Modelo de ElevenLabs para TTS streaming. "
            f"Default/env: ELEVENLABS_MODEL_ID o {DEFAULT_ELEVENLABS_MODEL_ID}."
        ),
    )
    parser.add_argument(
        "--elevenlabs-output-format",
        default=None,
        help=(
            "Formato de audio de ElevenLabs. "
            f"Default/env: ELEVENLABS_OUTPUT_FORMAT o {DEFAULT_ELEVENLABS_OUTPUT_FORMAT}."
        ),
    )
    parser.add_argument(
        "--elevenlabs-language-code",
        default=None,
        help=(
            "Codigo de idioma enviado a ElevenLabs. "
            f"Default/env: ELEVENLABS_LANGUAGE_CODE o {DEFAULT_ELEVENLABS_LANGUAGE_CODE}."
        ),
    )
    parser.add_argument(
        "--elevenlabs-stability",
        type=float,
        default=None,
        help="Voice setting opcional `stability` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-similarity-boost",
        type=float,
        default=None,
        help="Voice setting opcional `similarity_boost` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-style",
        type=float,
        default=None,
        help="Voice setting opcional `style` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-speed",
        type=float,
        default=None,
        help="Voice setting opcional `speed` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-use-speaker-boost",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activa o desactiva `use_speaker_boost` en ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-optimize-streaming-latency",
        type=int,
        choices=range(0, 5),
        default=None,
        metavar="{0,1,2,3,4}",
        help="Optimizacion de latencia streaming de ElevenLabs.",
    )
    parser.add_argument(
        "--llama-cpp-config",
        default=str(LLAMA_CPP_CONFIG_PATH),
        help="Ruta al config YAML usado para lanzar `llama-server`.",
    )
    return parser.parse_args()


def create_http_server(host, port, handler_class, max_port_tries=50):
    last_error = None
    for candidate_port in range(port, port + max_port_tries + 1):
        try:
            return ThreadingHTTPServer((host, candidate_port), handler_class)
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.EADDRINUSE:
                raise
    raise OSError(
        errno.EADDRINUSE,
        f"No se pudo encontrar un puerto libre desde {host}:{port} tras {max_port_tries + 1} intentos.",
    ) from last_error


def main():
    global COMMENTARY_CONFIG
    args = parse_args()
    COMMENTARY_CONFIG = commentary_config_from_args(args)
    server = create_http_server(args.host, args.port, InterfaceRequestHandler)
    bound_host, bound_port = server.server_address[:2]
    print(
        f"Interfaz disponible en http://{bound_host}:{bound_port} "
        f"(Python: {sys.executable})"
    )
    if COMMENTARY_CONFIG.backend == "llama_cpp":
        llama_cfg = COMMENTARY_CONFIG.llama_cpp_config or {}
        print(
            "Config local de llama.cpp: "
            f"{llama_cfg.get('config_path') or LLAMA_CPP_CONFIG_PATH}"
        )
    print(
        "Comentarios activados via fase CommentaryPhase del pipeline. "
        "Backend LLM: "
        f"{COMMENTARY_CONFIG.backend_display_name} "
        f"({COMMENTARY_CONFIG.model})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
