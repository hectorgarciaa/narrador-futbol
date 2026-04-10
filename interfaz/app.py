#!/usr/bin/env python3
"""
Interfaz web ligera para introducir alineaciones y lanzar `scripts/track.py`.
"""

from __future__ import annotations

import argparse
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
from football_ai.commentaries.deferred_media import (
    assemble_deferred_commentary_video,
)
from football_ai.commentaries.generator import OllamaCommentaryGenerator
from football_ai.commentaries.llama_cpp_backend import LlamaCppCommentaryGenerator
from football_ai.commentaries.server import (
    DEFAULT_SERVER_HOST as DEFAULT_COMMENTARY_HOST,
    DEFAULT_SERVER_PORT as DEFAULT_COMMENTARY_PORT,
    create_http_server,
)
from football_ai.commentaries.voice import CommentaryAudioPipeline, build_voice_synthesizer
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
OLLAMA_SERVE_LOG_PATH = COMMENTARY_CACHE_ROOT / "ollama_serve.log"
LLAMA_CPP_CONFIG_PATH = PROJECT_ROOT / "llama.cpp" / "config.yaml"
LLAMA_CPP_SERVER_LOG_PATH = COMMENTARY_CACHE_ROOT / "llama_cpp_server.log"
OLLAMA_STARTUP_TIMEOUT_SECONDS = 90.0
OLLAMA_STARTUP_POLL_SECONDS = 0.5

RUNS = {}
RUNS_LOCK = threading.Lock()
DEFAULT_COMMENTARY_MODE = "live"
COMMENTARY_MODE_CHOICES = {"live", "deferred"}
COMMENTARY_SERVER_MANAGER = None
STARTUP_INTRO_LOCK = threading.Lock()
STARTUP_INTRO_STATE = {
    "status": "idle",
    "started_at_utc": None,
    "finished_at_utc": None,
    "error": None,
    "thread_name": None,
}

APP_LIVE_COMMENTARY_ENV = "NARRADOR_APP_ENABLE_LIVE_COMMENTARY"
APP_COMMENTARY_SERVICE_URL_ENV = "NARRADOR_APP_COMMENTARY_SERVICE_URL"
APP_COMMENTARY_MANIFEST_PATH_ENV = "NARRADOR_APP_COMMENTARY_MANIFEST_PATH"
APP_COMMENTARY_AUDIO_DIR_ENV = "NARRADOR_APP_COMMENTARY_AUDIO_DIR"
APP_COMMENTARY_MODE_ENV = "NARRADOR_APP_COMMENTARY_MODE"
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


def get_startup_intro_state():
    with STARTUP_INTRO_LOCK:
        return dict(STARTUP_INTRO_STATE)


def update_startup_intro_state(**changes):
    with STARTUP_INTRO_LOCK:
        STARTUP_INTRO_STATE.update(changes)
        return dict(STARTUP_INTRO_STATE)


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


class CommentaryServerManager:
    def __init__(
        self,
        host=DEFAULT_COMMENTARY_HOST,
        port=DEFAULT_COMMENTARY_PORT,
        *,
        backend="ollama",
        model="gemma4:e2b",
        temperature=0.4,
        base_url=None,
        llama_cpp_config=None,
    ):
        self.host = str(host).strip() or DEFAULT_COMMENTARY_HOST
        self.port = int(port)
        self.backend = str(backend or "ollama").strip().lower() or "ollama"
        default_model = "gemma4-q4ks-text" if self.backend == "llama_cpp" else "gemma4:e2b"
        self.model = str(model).strip() or default_model
        self.temperature = float(temperature)
        self.base_url = str(base_url).strip() if base_url else None
        self.llama_cpp_config = dict(llama_cpp_config or {})
        self._start_lock = threading.Lock()
        self._ready_event = threading.Event()
        self._thread = None
        self._server = None
        self._start_error = None
        self._reused_external = False
        self._ollama_process = None
        self._ollama_log_handle = None
        self._ollama_managed_by_interface = False

    @property
    def service_url(self):
        return f"http://{self.host}:{self.port}"

    @property
    def backend_display_name(self):
        return "llama.cpp" if self.backend == "llama_cpp" else "Ollama"

    @property
    def managed_backend_log_path(self):
        if self.backend == "llama_cpp":
            return LLAMA_CPP_SERVER_LOG_PATH
        return OLLAMA_SERVE_LOG_PATH

    @property
    def llm_base_url(self):
        if self.backend == "llama_cpp":
            base_url = str(self.base_url).strip() if self.base_url else None
            if base_url:
                return base_url.rstrip("/")
            config_base_url = str(self.llama_cpp_config.get("base_url") or "").strip()
            if config_base_url:
                return config_base_url.rstrip("/")
            env_base_url = str(os.environ.get("LLAMA_CPP_BASE_URL") or "").strip()
            if not env_base_url:
                return "http://127.0.0.1:8001"
            if re.match(r"^https?://", env_base_url, flags=re.IGNORECASE):
                return env_base_url.rstrip("/")
            return f"http://{env_base_url}".rstrip("/")

        base_url = str(self.base_url).strip() if self.base_url else None
        if base_url:
            return base_url.rstrip("/")
        env_host = str(os.environ.get("OLLAMA_HOST") or "").strip()
        if not env_host:
            return "http://127.0.0.1:11434"
        if re.match(r"^https?://", env_host, flags=re.IGNORECASE):
            return env_host.rstrip("/")
        return f"http://{env_host}".rstrip("/")

    @property
    def ollama_managed_by_interface(self):
        return bool(self._ollama_managed_by_interface)

    def _health_url(self):
        return f"{self.service_url}/health"

    def _llm_health_url(self):
        if self.backend == "llama_cpp":
            return f"{self.llm_base_url}/v1/models"
        return f"{self.llm_base_url}/api/tags"

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

    def _probe_llm(self, timeout=1.5):
        try:
            with request.urlopen(self._llm_health_url(), timeout=timeout) as response:
                if response.status != 200:
                    return None
                payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict):
                    payload["base_url"] = self.llm_base_url
                    payload["backend"] = self.backend
                return payload
        except Exception:
            return None

    def _llm_target_is_local(self):
        parsed = urlparse(self.llm_base_url)
        host = str(parsed.hostname or "").strip().casefold()
        return host in {"", "127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def _close_ollama_log_handle(self):
        if self._ollama_log_handle is None:
            return
        try:
            self._ollama_log_handle.close()
        finally:
            self._ollama_log_handle = None

    def _spawn_ollama_process(self):
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            raise RuntimeError(
                "No se encontró el binario `ollama` en el PATH para lanzar `ollama serve`."
            )
        if self._ollama_process is not None and self._ollama_process.poll() is None:
            return
        self._close_ollama_log_handle()
        OLLAMA_SERVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._ollama_log_handle = open(
            OLLAMA_SERVE_LOG_PATH,
            "a",
            encoding="utf-8",
        )
        self._ollama_log_handle.write(
            f"\n[{now_iso()}] Lanzando `ollama serve` desde la interfaz.\n"
        )
        self._ollama_log_handle.flush()
        self._ollama_process = subprocess.Popen(
            [ollama_bin, "serve"],
            cwd=str(PROJECT_ROOT),
            stdout=self._ollama_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._ollama_managed_by_interface = True

    def _spawn_llama_cpp_process(self):
        executable_path = _resolve_optional_local_path(
            self.llama_cpp_config.get("config_dir") or (PROJECT_ROOT / "llama.cpp"),
            self.llama_cpp_config.get("executable_path"),
        )
        model_path = _resolve_optional_local_path(
            self.llama_cpp_config.get("config_dir") or (PROJECT_ROOT / "llama.cpp"),
            self.llama_cpp_config.get("model_path"),
        )
        if executable_path is None or not executable_path.exists():
            raise RuntimeError(
                "No se encontró el binario `llama-server`. "
                f"Revisa {self.llama_cpp_config.get('config_path') or LLAMA_CPP_CONFIG_PATH}."
            )
        if model_path is None or not model_path.exists():
            raise RuntimeError(
                "No se encontró el modelo GGUF para llama.cpp. "
                f"Esperado en {model_path}."
            )
        if self._ollama_process is not None and self._ollama_process.poll() is None:
            return

        parsed = urlparse(self.llm_base_url)
        host = str(parsed.hostname or self.llama_cpp_config.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        port = int(parsed.port or self.llama_cpp_config.get("port") or 8001)
        command = [
            str(executable_path),
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(port),
            "-a",
            self.model,
        ]

        ctx_size = self.llama_cpp_config.get("ctx_size")
        if ctx_size is not None:
            command.extend(["-c", str(int(ctx_size))])
        n_gpu_layers = self.llama_cpp_config.get("n_gpu_layers")
        if n_gpu_layers is not None:
            command.extend(["-ngl", str(int(n_gpu_layers))])
        reasoning = str(self.llama_cpp_config.get("reasoning") or "").strip()
        if reasoning:
            command.extend(["--reasoning", reasoning])
        if bool(self.llama_cpp_config.get("no_warmup", True)):
            command.append("--no-warmup")
        command.extend(self.llama_cpp_config.get("extra_args") or [])

        self._close_ollama_log_handle()
        LLAMA_CPP_SERVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._ollama_log_handle = open(
            LLAMA_CPP_SERVER_LOG_PATH,
            "a",
            encoding="utf-8",
        )
        self._ollama_log_handle.write(
            f"\n[{now_iso()}] Lanzando `llama-server` desde la interfaz.\n"
        )
        self._ollama_log_handle.write(
            f"[{now_iso()}] Comando: {' '.join(command)}\n"
        )
        self._ollama_log_handle.flush()
        self._ollama_process = subprocess.Popen(
            command,
            cwd=str(executable_path.parent),
            stdout=self._ollama_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self._ollama_managed_by_interface = True

    def _ensure_llm_ready(self, timeout=OLLAMA_STARTUP_TIMEOUT_SECONDS):
        payload = self._probe_llm(timeout=1.5)
        if payload is not None:
            return payload
        if not self._llm_target_is_local():
            return None

        if self.backend == "llama_cpp":
            self._spawn_llama_cpp_process()
        else:
            self._spawn_ollama_process()
        deadline = time.perf_counter() + float(timeout)
        while time.perf_counter() < deadline:
            payload = self._probe_llm(timeout=1.5)
            if payload is not None:
                return payload
            if self._ollama_process is not None and self._ollama_process.poll() is not None:
                raise RuntimeError(
                    f"`{self.backend_display_name}` terminó antes de exponer la API. "
                    f"Revisa el log en {self.managed_backend_log_path}."
                )
            time.sleep(OLLAMA_STARTUP_POLL_SECONDS)
        raise TimeoutError(
            f"La interfaz lanzó `{self.backend_display_name}`, pero la API no respondió "
            f"a tiempo en {self.llm_base_url}."
        )

    def build_commentary_generator(self):
        if self.backend == "llama_cpp":
            return LlamaCppCommentaryGenerator(
                model=self.model,
                temperature=self.temperature,
                base_url=self.llm_base_url,
            )
        return OllamaCommentaryGenerator(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )

    def start(self):
        with self._start_lock:
            if self._thread is not None or self._reused_external:
                return
            self._start_error = None
            self._ready_event.clear()
            external = self._probe_health()
            if external is not None:
                self._reused_external = True
                self._ready_event.set()
                return
            try:
                self._ensure_llm_ready()
            except Exception as exc:
                self._start_error = exc
                self._ready_event.set()
                return
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()

    def _serve(self):
        try:
            generator = self.build_commentary_generator()
            voice_synthesizer = build_voice_synthesizer()
            pipeline = CommentaryAudioPipeline(
                commentary_generator=generator,
                voice_synthesizer=voice_synthesizer,
            )
            pipeline.prepare()
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
        llm_payload = self._probe_llm(timeout=1.0)
        llm_status = "ready" if llm_payload is not None else "unavailable"
        if llm_status != "ready" and self._ollama_process is not None:
            if self._ollama_process.poll() is None:
                llm_status = "starting"
            elif self.ollama_managed_by_interface:
                llm_status = "failed"
        if self._reused_external:
            payload = self._probe_health(timeout=1.0) or {"status": "unavailable"}
            payload["managed_by_interface"] = False
            payload["service_url"] = self.service_url
            payload["llm_backend"] = self.backend
            payload["llm_status"] = llm_status
            payload["llm_base_url"] = self.llm_base_url
            payload["llm_managed_by_interface"] = self.ollama_managed_by_interface
            payload["llm_model"] = self.model
            if self.backend == "llama_cpp":
                payload["llama_cpp_config_path"] = str(
                    self.llama_cpp_config.get("config_path") or LLAMA_CPP_CONFIG_PATH
                )
            return payload
        if self._start_error is not None:
            return {
                "status": "failed",
                "service_url": self.service_url,
                "managed_by_interface": True,
                "error": str(self._start_error),
                "llm_backend": self.backend,
                "llm_status": llm_status,
                "llm_base_url": self.llm_base_url,
                "llm_managed_by_interface": self.ollama_managed_by_interface,
                "llm_model": self.model,
            }
        if self._thread is None:
            return {
                "status": "idle",
                "service_url": self.service_url,
                "managed_by_interface": True,
                "llm_backend": self.backend,
                "llm_status": llm_status,
                "llm_base_url": self.llm_base_url,
                "llm_managed_by_interface": self.ollama_managed_by_interface,
                "llm_model": self.model,
            }
        if not self._ready_event.is_set():
            return {
                "status": "starting",
                "service_url": self.service_url,
                "managed_by_interface": True,
                "llm_backend": self.backend,
                "llm_status": llm_status,
                "llm_base_url": self.llm_base_url,
                "llm_managed_by_interface": self.ollama_managed_by_interface,
                "llm_model": self.model,
            }
        payload = self._probe_health(timeout=1.0) or {"status": "unavailable"}
        payload["managed_by_interface"] = True
        payload["service_url"] = self.service_url
        payload["llm_backend"] = self.backend
        payload["llm_status"] = llm_status
        payload["llm_base_url"] = self.llm_base_url
        payload["llm_managed_by_interface"] = self.ollama_managed_by_interface
        payload["llm_model"] = self.model
        if self.backend == "llama_cpp":
            payload["llama_cpp_config_path"] = str(
                self.llama_cpp_config.get("config_path") or LLAMA_CPP_CONFIG_PATH
            )
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
        if self.ollama_managed_by_interface and self._ollama_process is not None:
            try:
                if self._ollama_process.poll() is None:
                    self._ollama_process.terminate()
                    self._ollama_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._ollama_process.kill()
                self._ollama_process.wait(timeout=5.0)
            finally:
                self._ollama_process = None
        self._close_ollama_log_handle()


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
    generator = COMMENTARY_SERVER_MANAGER.build_commentary_generator()
    voice_synthesizer = build_voice_synthesizer()
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


def start_startup_intro_preparation_background():
    current_state = get_startup_intro_state()
    if current_state.get("status") == "starting":
        return False

    def _worker():
        thread_name = threading.current_thread().name
        update_startup_intro_state(
            status="starting",
            started_at_utc=now_iso(),
            finished_at_utc=None,
            error=None,
            thread_name=thread_name,
        )
        try:
            prepare_startup_intro_commentary()
        except Exception as exc:  # pragma: no cover - defensivo
            update_startup_intro_state(
                status="failed",
                finished_at_utc=now_iso(),
                error=str(exc),
                thread_name=thread_name,
            )
            print(f"No se pudo precalentar el intro en segundo plano: {exc}")
        else:
            update_startup_intro_state(
                status="ready",
                finished_at_utc=now_iso(),
                error=None,
                thread_name=thread_name,
            )
            print(f"Intro precalentado en segundo plano en {STARTUP_INTRO_AUDIO_PATH}")

    worker = threading.Thread(
        target=_worker,
        name="startup-intro-warmup",
        daemon=True,
    )
    worker.start()
    return True


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


def resolve_result_video_path_from_status(status):
    if not isinstance(status, dict):
        return None
    commentary = dict(status.get("commentary") or {})
    commentary_mode = str(commentary.get("mode") or "").strip().lower()
    deferred_status = str(commentary.get("deferred_mux_status") or "").strip().lower()

    candidates = []
    if commentary_mode == "deferred":
        if deferred_status == "ready":
            candidates.append(commentary.get("deferred_video_path"))
        elif deferred_status == "failed":
            candidates.append(status.get("output_video_path"))
    elif str(status.get("status") or "").strip().lower() not in {"queued", "running"}:
        candidates.append(status.get("output_video_path"))

    for candidate in candidates:
        if not candidate:
            continue
        resolved_path = Path(candidate).expanduser().resolve()
        if resolved_path.exists() and resolved_path.is_file():
            return resolved_path
    return None


def attach_cached_intro_to_run(run_id, commentary_mode):
    run_paths = build_run_paths(run_id)
    run_paths["commentary_audio_dir"].mkdir(parents=True, exist_ok=True)
    cached_intro = read_startup_intro_commentary()
    if cached_intro is None:
        intro_state = get_startup_intro_state()
        intro_status = str(intro_state.get("status") or "pending").strip().lower()
        if intro_status == "ready":
            intro_status = "pending"
        if intro_status == "idle":
            start_startup_intro_preparation_background()
            intro_status = "starting"
        payload = {
            **initial_commentary_payload(run_id, run_paths, commentary_mode),
            "intro_status": "failed" if intro_status == "failed" else intro_status,
        }
        if intro_state.get("error"):
            payload["intro_error"] = str(intro_state["error"])
        return payload

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
    process_env = dict(os.environ)
    process_env.update(
        {
            APP_LIVE_COMMENTARY_ENV: "1",
            APP_COMMENTARY_SERVICE_URL_ENV: commentary_service_url(),
            APP_COMMENTARY_MANIFEST_PATH_ENV: str(run_paths["commentary_manifest_path"]),
            APP_COMMENTARY_AUDIO_DIR_ENV: str(run_paths["commentary_audio_dir"]),
            APP_COMMENTARY_MODE_ENV: str(commentary_mode),
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
        result_video_path = resolve_result_video_path_from_status(status)
        status["result_video_path"] = str(result_video_path) if result_video_path is not None else None
        status["result_video_url"] = (
            f"/api/runs/{run_id}/result-video"
            if result_video_path is not None
            else None
        )
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
        payload["startup_intro"] = get_startup_intro_state()
        cached_intro = read_startup_intro_commentary()
        payload["startup_intro"]["cache_ready"] = cached_intro is not None
        if cached_intro is not None:
            payload["startup_intro"]["generated_at_utc"] = cached_intro.get(
                "generated_at_utc"
            )
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
        result_video_path = resolve_result_video_path_from_status(status)
        if result_video_path is None:
            self._send_json(
                {"error": "El vídeo final todavía no está disponible."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        project_root = PROJECT_ROOT.resolve()
        if project_root not in result_video_path.parents and result_video_path != project_root:
            self._send_json(
                {"error": "Ruta de vídeo no permitida."},
                status=HTTPStatus.FORBIDDEN,
                include_body=include_body,
            )
            return
        self._send_file(result_video_path, include_body=include_body)

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
        default=0.4,
        help="Temperatura para el LLM de comentarios.",
    )
    parser.add_argument(
        "--commentary-base-url",
        default=None,
        help="URL base del backend LLM para el servidor de comentarios.",
    )
    parser.add_argument(
        "--llama-cpp-config",
        default=str(LLAMA_CPP_CONFIG_PATH),
        help="Ruta al config YAML usado para lanzar `llama-server`.",
    )
    return parser.parse_args()


def main():
    global COMMENTARY_SERVER_MANAGER
    args = parse_args()
    commentary_runtime = resolve_commentary_runtime_settings(args)
    COMMENTARY_SERVER_MANAGER = CommentaryServerManager(
        host=args.commentary_host,
        port=args.commentary_port,
        backend=commentary_runtime["backend"],
        model=commentary_runtime["model"],
        temperature=args.commentary_temperature,
        base_url=commentary_runtime["base_url"],
        llama_cpp_config=commentary_runtime["llama_cpp_config"],
    )
    server = ThreadingHTTPServer((args.host, args.port), InterfaceRequestHandler)
    warmup_started = start_startup_intro_preparation_background()
    print(
        f"Interfaz disponible en http://{args.host}:{args.port} "
        f"(Python: {sys.executable})"
    )
    print(
        "Servidor de comentarios disponible en "
        f"{COMMENTARY_SERVER_MANAGER.service_url}"
    )
    print(
        "Backend LLM de comentarios: "
        f"{COMMENTARY_SERVER_MANAGER.backend_display_name} "
        f"({COMMENTARY_SERVER_MANAGER.model})"
    )
    print(
        "API del backend LLM apuntando a "
        f"{COMMENTARY_SERVER_MANAGER.llm_base_url}"
    )
    if COMMENTARY_SERVER_MANAGER.backend == "llama_cpp":
        print(
            "Config local de llama.cpp: "
            f"{COMMENTARY_SERVER_MANAGER.llama_cpp_config.get('config_path') or LLAMA_CPP_CONFIG_PATH}"
        )
    if COMMENTARY_SERVER_MANAGER.ollama_managed_by_interface:
        print(
            f"La interfaz ha lanzado `{COMMENTARY_SERVER_MANAGER.backend_display_name}` "
            f"automaticamente. Log: {COMMENTARY_SERVER_MANAGER.managed_backend_log_path}"
        )
    cached_intro = read_startup_intro_commentary()
    if cached_intro is not None:
        print(f"Intro cacheado disponible en {STARTUP_INTRO_AUDIO_PATH}")
    if warmup_started:
        print("Precalentando intro y servidor de comentarios en segundo plano...")
    else:
        print("El precalentado del intro ya estaba en marcha.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        COMMENTARY_SERVER_MANAGER.shutdown()


if __name__ == "__main__":
    main()
