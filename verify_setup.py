#!/usr/bin/env python
"""
Script de verificacion del setup del proyecto.

Comprueba:
- version de Python y venv;
- dependencias principales;
- imports del paquete;
- config.yaml y estructura base;
- assets del runtime por defecto:
  - modelo YOLO principal;
  - video de prueba;
  - PathCRF si actions.enabled=true;
  - llama.cpp o endpoint remoto si commentary.enabled=true;
  - credenciales de ElevenLabs si generate_audio=true y tts_backend=elevenlabs.
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
from urllib import error, request


PROJECT_ROOT = Path(__file__).parent.resolve()
MIN_PYTHON = (3, 8)
TARGET_PYTHON = (3, 13, 7)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_info(text: str) -> None:
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def _format_version(version: Tuple[int, ...]) -> str:
    return ".".join(str(v) for v in version)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_placeholder(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    placeholder_tokens = (
        "your_",
        "<",
        ">",
        "changeme",
        "placeholder",
    )
    return any(token in text for token in placeholder_tokens)


def _load_env_file(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


DOTENV_VALUES = _load_env_file(PROJECT_ROOT / ".env")


def _env_value(name: str) -> str | None:
    return _clean_text(os.environ.get(name) or DOTENV_VALUES.get(name))


def _resolve_optional_local_path(base_dir: Path, value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _read_llama_cpp_config() -> Tuple[Dict[str, Any], Path, Path] | None:
    import yaml

    config_path = (PROJECT_ROOT / "external" / "llama.cpp" / "config.yaml").resolve()
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config invalido de llama.cpp: {config_path}")
    return payload, config_path.parent, config_path


def _check_http_health(base_url: str, timeout_s: float = 2.0) -> bool:
    try:
        with request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=timeout_s) as response:
            return response.status == 200
    except Exception:
        return False


def check_python_version() -> bool:
    current = sys.version_info[:3]
    current_str = _format_version(current)
    min_str = _format_version(MIN_PYTHON)
    target_str = _format_version(TARGET_PYTHON)

    if current < MIN_PYTHON:
        print_error(f"Python {current_str} - se requiere >= {min_str}")
        return False

    print_success(f"Python {current_str} (minimo soportado: {min_str})")
    if current != TARGET_PYTHON:
        print_warning(
            f"La version objetivo del repo es {target_str}. "
            f"La actual es {current_str}."
        )
    else:
        print_success(f"Coincide con la version objetivo del repo: {target_str}")
    return True


def check_venv() -> bool:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print_success(f"Entorno virtual activo: {sys.prefix}")
        return True
    print_warning(
        "No se detecta venv activo. Se recomienda usar `.venv` antes de ejecutar el pipeline."
    )
    return True


def check_dependencies() -> bool:
    required = {
        "torch": "torch",
        "ultralytics": "ultralytics",
        "supervision": "supervision",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
    }
    all_ok = True
    for module_name, package_name in required.items():
        try:
            module = import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print_success(f"{package_name:25} {version}")
        except ImportError:
            print_error(f"{package_name:25} NOT INSTALLED")
            all_ok = False
    return all_ok


def check_football_ai_imports() -> bool:
    modules = [
        "football_ai.core.config",
        "football_ai.pipeline.pipeline",
        "football_ai.tracking.tracker",
        "football_ai.actions.rolling",
        "football_ai.commentaries.phase",
        "football_ai.visualization.drawer",
    ]
    all_ok = True
    for module_path in modules:
        try:
            import_module(module_path)
            print_success(module_path)
        except Exception as exc:
            print_error(f"{module_path}: {str(exc)[:120]}")
            all_ok = False
    return all_ok


def check_torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print_error("torch no esta instalado")
        return False

    print_info(f"Version de torch: {torch.__version__}")
    if torch.cuda.is_available():
        print_success(
            f"CUDA disponible (version {torch.version.cuda}, devices={torch.cuda.device_count()})"
        )
    else:
        print_warning("CUDA no disponible. El proyecto puede correr en CPU, pero mas lento.")
    return True


def check_config_files() -> bool:
    files_required = {
        "config.yaml": PROJECT_ROOT / "config.yaml",
        "requirements.txt": PROJECT_ROOT / "requirements.txt",
        "pyproject.toml": PROJECT_ROOT / "pyproject.toml",
    }
    all_ok = True
    for label, path in files_required.items():
        if path.exists():
            print_success(label)
        else:
            print_error(f"{label} - NO ENCONTRADO")
            all_ok = False

    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        print_success(".env.example")
    else:
        print_warning(".env.example no existe")

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print_success(".env")
    else:
        print_warning(".env no existe. Es opcional salvo para algunos backends externos.")
    return all_ok


def check_data_structure() -> bool:
    paths = {
        "football_ai/": PROJECT_ROOT / "football_ai",
        "scripts/": PROJECT_ROOT / "scripts",
        "models/": PROJECT_ROOT / "models",
        "data/": PROJECT_ROOT / "data",
        "data/partidoPrueba/": PROJECT_ROOT / "data" / "partidoPrueba",
    }
    all_ok = True
    for label, path in paths.items():
        if path.exists():
            print_success(label)
        else:
            print_warning(f"{label} no existe")
            all_ok = False
    return all_ok


def check_optional_env() -> bool:
    roboflow_keys = [
        "ROBOFLOW_API_KEY",
        "ROBOFLOW_WORKSPACE",
        "ROBOFLOW_PROJECT",
    ]
    missing = [key for key in roboflow_keys if _is_placeholder(_env_value(key))]
    if missing:
        print_warning(
            "Variables de Roboflow ausentes o placeholder. "
            "Solo hacen falta para descargar datasets."
        )
    else:
        print_success("Variables de Roboflow listas para descarga de datasets")

    llama_env = _env_value("LLAMA_CPP_BASE_URL")
    if llama_env:
        print_success(f"LLAMA_CPP_BASE_URL configurado: {llama_env}")
    else:
        print_info("LLAMA_CPP_BASE_URL no definido")
    return True


def _check_default_video_paths(config: Any) -> bool:
    all_ok = True
    video_prueba_medio = config.get_path("paths", "data", "video_prueba_medio")
    if video_prueba_medio.exists():
        print_success(f"Video de prueba medio: {video_prueba_medio}")
    else:
        print_error(f"Falta video_prueba_medio: {video_prueba_medio}")
        all_ok = False

    video_source_key = config.get("tracking", "lineup_spec", default={}).get("video_source")
    if video_source_key:
        resolved = config.get_path("paths", "data", str(video_source_key))
        if resolved.exists():
            print_success(f"tracking.lineup_spec.video_source -> {resolved}")
        else:
            print_error(
                f"Falta el video configurado en tracking.lineup_spec.video_source "
                f"({video_source_key} -> {resolved})"
            )
            all_ok = False
    return all_ok


def _check_pathcrf(actions_conf: Mapping[str, Any]) -> bool:
    repo_path = (PROJECT_ROOT / str(actions_conf.get("repo_path", "external/pathcrf"))).resolve()
    trial = int(actions_conf.get("trial", 120))
    model_file = str(actions_conf.get("model_file", "state_dict_best_acc.pt"))
    trial_dir = repo_path / "saved" / f"{trial:03d}"
    args_json = trial_dir / "args.json"
    checkpoint = trial_dir / "model" / model_file

    all_ok = True
    if not repo_path.exists():
        print_error(f"Falta repo PathCRF: {repo_path}")
        return False
    print_success(f"Repo PathCRF: {repo_path}")

    if not trial_dir.exists():
        print_error(f"Falta el trial PathCRF: {trial_dir}")
        all_ok = False
    else:
        print_success(f"Trial PathCRF: {trial_dir}")

    if not args_json.exists():
        print_error(f"Falta args.json de PathCRF: {args_json}")
        all_ok = False
    else:
        print_success(f"args.json de PathCRF: {args_json}")

    if not checkpoint.exists():
        print_error(f"Falta checkpoint de PathCRF: {checkpoint}")
        all_ok = False
    else:
        print_success(f"Checkpoint de PathCRF: {checkpoint}")

    return all_ok


def _check_llama_cpp(commentary_conf: Mapping[str, Any]) -> bool:
    llm_base_url = _clean_text(commentary_conf.get("llm_base_url")) or _env_value("LLAMA_CPP_BASE_URL")
    if llm_base_url:
        print_success(f"Backend LLM configurado por URL: {llm_base_url}")
        if _check_http_health(llm_base_url):
            print_success("El endpoint LLM responde en /v1/models")
        else:
            print_warning(
                "El endpoint LLM no responde ahora mismo. "
                "Puede ser normal si aun no esta levantado."
            )
        return True

    cfg = _read_llama_cpp_config()
    if cfg is None:
        print_error(
            "Comentario habilitado pero falta `external/llama.cpp/config.yaml` "
            "y tampoco hay `commentary.llm_base_url` ni `LLAMA_CPP_BASE_URL`."
        )
        return False

    payload, config_dir, config_path = cfg
    runtime_cfg = dict(payload.get("runtime") or {})
    server_cfg = dict(payload.get("server") or {})
    model_cfg = dict(payload.get("model") or {})

    if not bool(runtime_cfg.get("enabled", True)):
        print_error(
            f"{config_path} existe pero `runtime.enabled=false` y no hay endpoint remoto configurado."
        )
        return False

    executable = _resolve_optional_local_path(
        config_dir,
        server_cfg.get("executable") or "./llama-server",
    )
    model_path = _resolve_optional_local_path(
        config_dir,
        model_cfg.get("path"),
    )
    host = str(server_cfg.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(server_cfg.get("port") or 8001)
    base_url = f"http://{host}:{port}"

    print_success(f"Config local de llama.cpp: {config_path}")

    all_ok = True
    if executable is None or not executable.exists():
        print_error(f"Binario llama-server no encontrado: {executable}")
        all_ok = False
    else:
        print_success(f"Binario llama-server: {executable}")

    if model_path is None or not model_path.exists():
        print_error(f"Modelo GGUF no encontrado: {model_path}")
        all_ok = False
    else:
        print_success(f"Modelo GGUF: {model_path}")

    if _check_http_health(base_url):
        print_success(f"llama.cpp responde en {base_url}/v1/models")
    else:
        print_info(
            f"llama.cpp no responde ahora mismo en {base_url}. "
            "No es fallo si el launcher debe arrancarlo durante la ejecucion."
        )

    return all_ok


def _check_commentary_env(commentary_conf: Mapping[str, Any]) -> bool:
    if not bool(commentary_conf.get("generate_audio", True)):
        print_info("Audio de comentarios desactivado")
        return True

    tts_backend = str(commentary_conf.get("tts_backend", "xtts")).strip().lower()
    print_info(f"TTS backend configurado: {tts_backend}")

    if tts_backend != "elevenlabs":
        print_info("No se requieren credenciales extra de env para este backend TTS")
        return True

    api_key = _clean_text(commentary_conf.get("elevenlabs_api_key")) or _env_value("ELEVENLABS_API_KEY")
    voice_id = _clean_text(commentary_conf.get("elevenlabs_voice_id")) or _env_value("ELEVENLABS_VOICE_ID")
    female_voice_id = _clean_text(commentary_conf.get("elevenlabs_female_voice_id")) or _env_value("ELEVENLABS_FEMALE_VOICE_ID")
    alternate_voices = bool(commentary_conf.get("alternate_voices", False))

    all_ok = True
    if _is_placeholder(api_key):
        print_error("Falta ELEVENLABS_API_KEY para el backend elevenlabs")
        all_ok = False
    else:
        print_success("ELEVENLABS_API_KEY configurado")

    if _is_placeholder(voice_id):
        print_error("Falta ELEVENLABS_VOICE_ID para el backend elevenlabs")
        all_ok = False
    else:
        print_success("ELEVENLABS_VOICE_ID configurado")

    if alternate_voices:
        if _is_placeholder(female_voice_id):
            print_warning(
                "alternate_voices=true pero falta ELEVENLABS_FEMALE_VOICE_ID. "
                "La voz femenina no estara disponible."
            )
        else:
            print_success("ELEVENLABS_FEMALE_VOICE_ID configurado")

    return all_ok


def check_runtime_assets() -> bool:
    try:
        from football_ai.core.config import get_config
        from football_ai.positions import validate_lineup_payload
    except Exception as exc:
        print_error(f"No se pudo importar la configuracion del proyecto: {exc}")
        return False

    try:
        config = get_config()
    except Exception as exc:
        print_error(f"No se pudo cargar config.yaml: {exc}")
        return False

    print_success("config.yaml carga correctamente")
    all_ok = True

    model_path = config.get_path("paths", "models", "modelo_base")
    if model_path.exists():
        print_success(f"Modelo default de tracking: {model_path}")
    else:
        print_error(f"Falta el modelo default de tracking: {model_path}")
        all_ok = False

    all_ok = _check_default_video_paths(config) and all_ok

    lineup_spec = config.get("tracking", "lineup_spec", default=None)
    if isinstance(lineup_spec, dict):
        try:
            validate_lineup_payload(lineup_spec)
            print_success("tracking.lineup_spec valido")
        except Exception as exc:
            print_error(f"tracking.lineup_spec invalido: {exc}")
            all_ok = False
    elif lineup_spec:
        lineup_path = (config.project_root / str(lineup_spec)).resolve()
        if lineup_path.exists():
            print_success(f"tracking.lineup_spec: {lineup_path}")
        else:
            print_error(f"Falta tracking.lineup_spec: {lineup_path}")
            all_ok = False
    else:
        print_warning("tracking.lineup_spec no esta definido")

    actions_conf = dict(config.actions or {})
    if bool(actions_conf.get("enabled", False)):
        print_info("actions.enabled=true: se verificara PathCRF")
        all_ok = _check_pathcrf(actions_conf) and all_ok
    else:
        print_info("actions.enabled=false")

    commentary_conf = dict(config.commentary or {})
    if bool(commentary_conf.get("enabled", False)):
        print_info("commentary.enabled=true: se verificara el backend LLM/TTS")
        all_ok = _check_llama_cpp(commentary_conf) and all_ok
        all_ok = _check_commentary_env(commentary_conf) and all_ok
    else:
        print_info("commentary.enabled=false")

    return all_ok


def main() -> int:
    print_header("VERIFICACION DEL PROYECTO NARRADOR-FUTBOL")

    checks: List[Tuple[str, Any]] = [
        ("Python", check_python_version),
        ("Entorno virtual", check_venv),
        ("Archivos base", check_config_files),
        ("Estructura de directorios", check_data_structure),
        ("Dependencias principales", check_dependencies),
        ("PyTorch y CUDA", check_torch_cuda),
        ("Imports de football_ai", check_football_ai_imports),
        ("Variables de entorno opcionales", check_optional_env),
        ("Runtime por defecto", check_runtime_assets),
    ]

    results: Dict[str, bool] = {}
    for check_name, check_func in checks:
        print_header(f"-> {check_name}")
        try:
            results[check_name] = bool(check_func())
        except Exception as exc:
            print_error(f"Error al ejecutar la verificacion: {str(exc)[:160]}")
            results[check_name] = False

    print_header("RESUMEN")
    passed = sum(1 for value in results.values() if value)
    total = len(results)

    for check_name, result in results.items():
        symbol = f"{Colors.GREEN}✓{Colors.RESET}" if result else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{symbol} {check_name}")

    print(f"\n{Colors.BOLD}Checks correctos: {passed}/{total}{Colors.RESET}\n")

    if all(results.values()):
        print_success("El setup minimo del runtime por defecto esta listo.")
        print_info("Siguiente paso recomendado: python scripts/track.py video_prueba_medio")
        return 0

    print_error("Hay checks fallidos. Revisa los mensajes anteriores.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
