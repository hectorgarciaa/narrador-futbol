from __future__ import annotations

import subprocess
import time
from pathlib import Path
from urllib import error, request

from football_ai.core import get_config, get_logger

logger = get_logger(__name__)


def _resolve_relative(base_dir, value):
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return path


def _check_server_ready(base_url, timeout_s=60.0, poll_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with request.urlopen(f"{base_url}/v1/models", timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def ensure_llama_server_running() -> str | None:
    config = get_config()
    payload = dict(config.get("llama_cpp", default={}) or {})
    if not payload:
        logger.warning("Bloque `llama_cpp` no encontrado en config.yaml")
        return None
    config_dir = config.project_root
    config_path = config.project_root / "config.yaml"
    server_cfg = dict(payload.get("server") or {})
    model_cfg = dict(payload.get("model") or {})
    runtime_cfg = dict(payload.get("runtime") or {})

    if not bool(runtime_cfg.get("enabled", True)):
        logger.info("llama.cpp runtime desactivado en config: %s", config_path)
        return None

    executable = _resolve_relative(config_dir, server_cfg.get("executable") or "./llama-server")
    model_path = _resolve_relative(config_dir, model_cfg.get("path"))
    host = str(server_cfg.get("host") or "127.0.0.1").strip()
    port = int(server_cfg.get("port") or 8001)
    base_url = f"http://{host}:{port}"
    alias = str(server_cfg.get("alias") or "gemma4-q4ks-text").strip()
    ctx_size = int(server_cfg.get("ctx_size", 512))
    n_gpu_layers = int(server_cfg.get("n_gpu_layers", 999))

    if _check_server_ready(base_url, timeout_s=2.0, poll_s=0.5):
        logger.info("llama-server ya corriendo en %s (config: %s)", base_url, config_path)
        return base_url

    if executable is None or not Path(executable).exists():
        logger.warning("llama-server no encontrado en %s", executable)
        return None

    if model_path is None or not Path(model_path).exists():
        logger.warning("Modelo GGUF no encontrado en %s", model_path)
        return None

    cmd = [
        str(executable),
        "-m", str(model_path),
        "--alias", alias,
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "-ngl", str(n_gpu_layers),
    ]

    reasoning = str(server_cfg.get("reasoning") or "off").strip()
    if reasoning == "off":
        cmd.append("--reasoning")
        cmd.append("off")

    extra_args = server_cfg.get("extra_args") or runtime_cfg.get("extra_args") or []
    for arg in extra_args:
        cmd.append(str(arg))

    logger.info("Arrancando llama-server con config %s: %s", config_path, " ".join(cmd))
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        logger.error("No se pudo arrancar llama-server: %s", exc)
        return None

    if _check_server_ready(base_url, timeout_s=120.0, poll_s=2.0):
        logger.info("llama-server listo en %s", base_url)
        return base_url

    logger.error("llama-server no respondio tras 120s en %s", base_url)
    return None
