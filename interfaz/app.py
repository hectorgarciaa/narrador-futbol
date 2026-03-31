#!/usr/bin/env python3
"""
Interfaz web ligera para introducir alineaciones y lanzar `scripts/track.py`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import get_config
from football_ai.commentaries.deferred_media import (
    assemble_deferred_commentary_video,
)
from football_ai.commentaries.generator import OllamaCommentaryGenerator
from football_ai.commentaries.server import (
    DEFAULT_SERVER_HOST as DEFAULT_COMMENTARY_HOST,
    DEFAULT_SERVER_PORT as DEFAULT_COMMENTARY_PORT,
    create_http_server,
)
from football_ai.commentaries.voice import CommentaryAudioPipeline, XTTSVoiceSynthesizer
from football_ai.positions import (
    LineupSpecError,
    get_formation_catalog,
    sanitize_video_stem,
    validate_lineup_payload,
)
from football_ai.tracking_cli.paths import (
    build_output_video_path,
    resolve_video_path,
)


STATIC_ROOT = Path(__file__).resolve().parent / "static"
RUNS_ROOT = PROJECT_ROOT / "output" / "interfaz" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
COMMENTARY_CACHE_ROOT = PROJECT_ROOT / "output" / "interfaz" / "commentary_cache"
COMMENTARY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
STARTUP_INTRO_AUDIO_PATH = COMMENTARY_CACHE_ROOT / "startup_intro.wav"
STARTUP_INTRO_META_PATH = COMMENTARY_CACHE_ROOT / "startup_intro.json"

RUNS = {}
RUNS_LOCK = threading.Lock()
DEFAULT_COMMENTARY_MODE = "live"
COMMENTARY_MODE_CHOICES = {"live", "deferred"}
COMMENTARY_SERVER_MANAGER = None


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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


class CommentaryServerManager:
    def __init__(
        self,
        host=DEFAULT_COMMENTARY_HOST,
        port=DEFAULT_COMMENTARY_PORT,
        *,
        model="qwen3:1.7b",
        temperature=0.4,
        base_url=None,
    ):
        self.host = str(host).strip() or DEFAULT_COMMENTARY_HOST
        self.port = int(port)
        self.model = str(model).strip() or "qwen3:1.7b"
        self.temperature = float(temperature)
        self.base_url = str(base_url).strip() if base_url else None
        self._start_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._thread = None
        self._server = None
        self._start_error = None
        self._reused_external = False

    @property
    def service_url(self):
        return f"http://{self.host}:{self.port}"

    def _health_url(self):
        return f"{self.service_url}/health"

    def _probe_health(self, timeout=1.5):
        try:
            with request.urlopen(self._health_url(), timeout=timeout) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
                payload["service_url"] = self.service_url
                return payload
        except Exception:
            return None

    def start(self):
        with self._start_lock:
            if self._thread is not None or self._reused_external:
                return
            external = self._probe_health()
            if external is not None:
                self._reused_external = True
                self._ready_event.set()
                return
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()

    def _serve(self):
        try:
            generator = OllamaCommentaryGenerator(
                model=self.model,
                temperature=self.temperature,
                base_url=self.base_url,
            )
            voice_synthesizer = XTTSVoiceSynthesizer()
            pipeline = CommentaryAudioPipeline(
                commentary_generator=generator,
                voice_synthesizer=voice_synthesizer,
            )
            voice_synthesizer.prepare()
            self._server = create_http_server(
                commentary_generator=generator,
                audio_pipeline=pipeline,
                host=self.host,
                port=self.port,
                text_only=False,
            )
            self._ready_event.set()
            self._server.serve_forever()
        except Exception as exc:  # pragma: no cover - defensivo
            self._start_error = exc
            self._ready_event.set()

    def wait_until_ready(self, timeout=240.0):
        self.start()
        if self._reused_external:
            payload = self._probe_health(timeout=2.0)
            if payload is None:
                raise RuntimeError("El servidor externo de comentarios no responde.")
            return payload
        if not self._ready_event.wait(timeout):
            raise TimeoutError(
                "El servidor de comentarios no estuvo listo a tiempo."
            )
        if self._start_error is not None:
            raise RuntimeError(
                f"No se pudo arrancar el servidor de comentarios: {self._start_error}"
            ) from self._start_error
        payload = self._probe_health(timeout=2.0)
        if payload is None:
            raise RuntimeError("El servidor de comentarios arrancó pero no responde.")
        return payload

    def health_payload(self):
        if self._reused_external:
            payload = self._probe_health(timeout=1.0) or {"status": "unavailable"}
            payload["managed_by_interface"] = False
            payload["service_url"] = self.service_url
            return payload
        if self._thread is None:
            return {
                "status": "idle",
                "service_url": self.service_url,
                "managed_by_interface": True,
            }
        if not self._ready_event.is_set():
            return {
                "status": "starting",
                "service_url": self.service_url,
                "managed_by_interface": True,
            }
        if self._start_error is not None:
            return {
                "status": "failed",
                "service_url": self.service_url,
                "managed_by_interface": True,
                "error": str(self._start_error),
            }
        payload = self._probe_health(timeout=1.0) or {"status": "unavailable"}
        payload["managed_by_interface"] = True
        payload["service_url"] = self.service_url
        return payload

    def process_payload(self, payload, timeout=240.0):
        self.wait_until_ready(timeout=timeout)
        raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.service_url}/api/commentaries",
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                return int(response.status), json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed_body = json.loads(body)
            except Exception:
                parsed_body = {"error": body or str(exc)}
            return int(exc.code), parsed_body

    def shutdown(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def build_startup_intro_event():
    return {
        "action": "intro",
        "event_time_s": 0.0,
        "player_name": "",
        "player_position": "",
    }


def build_startup_intro_compat_event():
    return {
        "action": "intro",
        "event_time_s": 0.0,
        # Compatibilidad con servidores viejos que seguian validando
        # `player_name` y `player_position` incluso en intros.
        "player_name": "_",
        "player_position": "_",
    }


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


def _fallback_startup_intro_text():
    return (
        "Bienvenidos, ya esta todo preparado para disfrutar de un partido que "
        "promete emociones fuertes."
    )


def generate_startup_intro_commentary_locally():
    global COMMENTARY_SERVER_MANAGER
    if COMMENTARY_SERVER_MANAGER is None:
        raise RuntimeError("El servidor de comentarios no esta configurado.")

    event = build_startup_intro_event()
    generator = OllamaCommentaryGenerator(
        model=COMMENTARY_SERVER_MANAGER.model,
        temperature=COMMENTARY_SERVER_MANAGER.temperature,
        base_url=COMMENTARY_SERVER_MANAGER.base_url,
    )
    voice_synthesizer = XTTSVoiceSynthesizer()
    voice_synthesizer.prepare()
    pipeline = CommentaryAudioPipeline(
        commentary_generator=generator,
        voice_synthesizer=voice_synthesizer,
    )

    try:
        result = pipeline.generate_to_file(
            event,
            audio_path=STARTUP_INTRO_AUDIO_PATH,
        )
        if _looks_like_bad_intro_commentary(result.commentary):
            raise RuntimeError(
                "El modelo devolvio un intro con placeholders o plantilla legacy."
            )
        return {
            "commentary": result.commentary,
            "audio_path": str(result.audio_path),
            "model": result.commentary_result.model,
            "llm_seconds": round(result.llm_seconds or 0.0, 3),
            "tts_seconds": round(result.tts_seconds or 0.0, 3),
            "total_seconds": round(result.total_seconds or 0.0, 3),
        }
    except Exception:
        fallback_text = _fallback_startup_intro_text()
        tts_start = time.perf_counter()
        audio_path = voice_synthesizer.synthesize_to_file(
            fallback_text,
            STARTUP_INTRO_AUDIO_PATH,
        )
        tts_seconds = time.perf_counter() - tts_start
        return {
            "commentary": fallback_text,
            "audio_path": str(audio_path),
            "model": f"{COMMENTARY_SERVER_MANAGER.model}:intro-fallback",
            "llm_seconds": 0.0,
            "tts_seconds": round(tts_seconds, 3),
            "total_seconds": round(tts_seconds, 3),
        }


def prepare_startup_intro_commentary():
    global COMMENTARY_SERVER_MANAGER
    if COMMENTARY_SERVER_MANAGER is None:
        raise RuntimeError("El servidor de comentarios no esta configurado.")

    payload = {
        "event": build_startup_intro_event(),
        "audio_out": str(STARTUP_INTRO_AUDIO_PATH),
    }
    status_code, response_payload = COMMENTARY_SERVER_MANAGER.process_payload(
        payload,
        timeout=300.0,
    )
    if status_code != int(HTTPStatus.OK):
        error_text = str(response_payload.get("error") or "")
        needs_legacy_intro_retry = (
            "`player_name` no puede ir vacio." in error_text
            or "`player_position` no puede ir vacio." in error_text
        )
        if needs_legacy_intro_retry:
            response_payload = generate_startup_intro_commentary_locally()
            status_code = int(HTTPStatus.OK)
    if status_code != int(HTTPStatus.OK):
        raise RuntimeError(
            response_payload.get("error") or "No se pudo generar el intro inicial."
        )
    if _looks_like_bad_intro_commentary(response_payload.get("commentary")):
        response_payload = generate_startup_intro_commentary_locally()

    intro_payload = {
        "generated_at_utc": now_iso(),
        "event": build_startup_intro_event(),
        **response_payload,
    }
    write_json(STARTUP_INTRO_META_PATH, intro_payload)
    return intro_payload


def read_startup_intro_commentary():
    payload = read_json(STARTUP_INTRO_META_PATH, default=None)
    if payload is None:
        return None
    audio_path = payload.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        return None
    return payload


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


def resolve_tracking_output_video_path(video_source):
    config = get_config()
    resolved_video_path, _ = resolve_video_path(config, video_source)
    return Path(build_output_video_path(config, resolved_video_path)).resolve()


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


def commentary_service_url():
    global COMMENTARY_SERVER_MANAGER
    if COMMENTARY_SERVER_MANAGER is None:
        return f"http://{DEFAULT_COMMENTARY_HOST}:{DEFAULT_COMMENTARY_PORT}"
    return COMMENTARY_SERVER_MANAGER.service_url


def initial_commentary_payload(run_id, run_paths, commentary_mode):
    return {
        "mode": commentary_mode,
        "service_url": commentary_service_url(),
        "manifest_path": str(run_paths["commentary_manifest_path"]),
        "events_api_path": f"/api/runs/{run_id}/commentary-events",
        "audio_dir": str(run_paths["commentary_audio_dir"]),
        "commentary_track_path": str(run_paths["commentary_track_audio_path"]),
        "intro_status": "pending",
        "deferred_mux_status": "pending",
    }


def attach_cached_intro_to_run(run_id, commentary_mode):
    run_paths = build_run_paths(run_id)
    run_paths["commentary_audio_dir"].mkdir(parents=True, exist_ok=True)
    cached_intro = read_startup_intro_commentary()
    if cached_intro is None:
        raise RuntimeError(
            "El intro inicial no esta preparado todavia. "
            "Espera a que termine de arrancar la interfaz."
        )

    cached_audio_path = Path(cached_intro["audio_path"]).expanduser().resolve()
    shutil.copy2(cached_audio_path, run_paths["commentary_intro_audio_path"])

    intro_payload = {
        **cached_intro,
        "copied_at_utc": now_iso(),
        "audio_path": str(run_paths["commentary_intro_audio_path"]),
    }
    write_json(run_paths["commentary_intro_meta_path"], intro_payload)
    append_jsonl(
        run_paths["commentary_manifest_path"],
        {
            "generated_at_utc": intro_payload.get("generated_at_utc") or now_iso(),
            "event": intro_payload.get("event") or build_startup_intro_event(),
            "mode": commentary_mode,
            "metadata": {
                "run_id": str(run_id),
                "source": "interfaz",
                "event_kind": "intro",
                "prebuilt_on_interface_startup": True,
            },
            "commentary": intro_payload.get("commentary"),
            "audio_path": str(run_paths["commentary_intro_audio_path"]),
            "model": intro_payload.get("model"),
            "tts_model": intro_payload.get("tts_model"),
            "text_only": False,
            "llm_seconds": intro_payload.get("llm_seconds"),
            "tts_seconds": intro_payload.get("tts_seconds"),
            "total_seconds": intro_payload.get("total_seconds"),
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
        "llm_seconds": intro_payload.get("llm_seconds"),
        "tts_seconds": intro_payload.get("tts_seconds"),
        "total_seconds": intro_payload.get("total_seconds"),
    }


def initial_status_payload(run_id, lineup_spec, run_paths, commentary_mode):
    teams = lineup_spec["teams"]
    video_source = str(lineup_spec.get("video_source") or "").strip()
    video_stem = sanitize_video_stem(video_source) if video_source else None
    output_video_path = None
    if video_source:
        try:
            output_video_path = str(resolve_tracking_output_video_path(video_source))
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
    ]

    log_file = open(run_paths["log_path"], "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
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

        try:
            assembly_result = assemble_deferred_commentary_video(
                run_paths["commentary_manifest_path"],
                output_video_path,
                run_paths["commentary_track_audio_path"],
                output_video_path,
            )
            update_commentary_payload(
                run_id,
                deferred_mux_status="ready",
                commentary_track_path=str(assembly_result.commentary_track_path),
                deferred_video_path=str(assembly_result.output_video_path),
                deferred_event_count=int(assembly_result.event_count),
                deferred_video_duration_seconds=round(
                    float(assembly_result.video_duration_seconds),
                    3,
                ),
            )
            update_status_file(
                run_id,
                output_video_path=str(assembly_result.output_video_path),
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
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if include_body:
            self.wfile.write(content)

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
        self._send_json(status, include_body=include_body)

    def _handle_get_commentary_service(self, include_body=True):
        global COMMENTARY_SERVER_MANAGER
        if COMMENTARY_SERVER_MANAGER is None:
            payload = {
                "status": "unavailable",
                "service_url": commentary_service_url(),
            }
        else:
            payload = COMMENTARY_SERVER_MANAGER.health_payload()
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
            run_id = uuid.uuid4().hex[:12]
            status = launch_tracking_process(
                run_id,
                lineup_spec,
                commentary_mode=commentary_mode,
            )
            try:
                commentary_payload = attach_cached_intro_to_run(
                    run_id,
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
        default="qwen3:1.7b",
        help="Modelo Ollama por defecto para comentarios.",
    )
    parser.add_argument(
        "--commentary-temperature",
        type=float,
        default=0.4,
        help="Temperatura para el LLM de comentarios.",
    )
    parser.add_argument(
        "--commentary-base-url",
        default=None,
        help="URL base de Ollama para el servidor de comentarios.",
    )
    return parser.parse_args()


def main():
    global COMMENTARY_SERVER_MANAGER
    args = parse_args()
    COMMENTARY_SERVER_MANAGER = CommentaryServerManager(
        host=args.commentary_host,
        port=args.commentary_port,
        model=args.commentary_model,
        temperature=args.commentary_temperature,
        base_url=args.commentary_base_url,
    )
    COMMENTARY_SERVER_MANAGER.start()
    startup_intro_error = None
    try:
        prepare_startup_intro_commentary()
    except Exception as exc:  # pragma: no cover - defensivo
        startup_intro_error = exc
    server = ThreadingHTTPServer((args.host, args.port), InterfaceRequestHandler)
    print(
        f"Interfaz disponible en http://{args.host}:{args.port} "
        f"(Python: {sys.executable})"
    )
    print(
        "Servidor de comentarios disponible en "
        f"{COMMENTARY_SERVER_MANAGER.service_url}"
    )
    if startup_intro_error is None:
        print(f"Intro precalentado en {STARTUP_INTRO_AUDIO_PATH}")
    else:
        print(f"No se pudo precalentar el intro: {startup_intro_error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        COMMENTARY_SERVER_MANAGER.shutdown()


if __name__ == "__main__":
    main()
