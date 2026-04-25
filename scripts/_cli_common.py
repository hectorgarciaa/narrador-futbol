import argparse
import re
from datetime import datetime
from pathlib import Path

from football_ai.core import get_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_video_model_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("video_path", help="Shortcut de config paths.data o ruta desde la raiz")
    parser.add_argument("model_path", help="Shortcut de config paths.models o ruta desde la raiz")
    parser.add_argument("--config", default=None, help="Ruta alternativa a config.yaml")
    parser.add_argument("--max-frames", type=int, default=None, help="Limita frames procesados")
    return parser


def load_config(config_path=None):
    return get_config(config_path)


def resolve_config_or_project_path(config, group, value):
    group_paths = config.paths.get(group, {})
    if value in group_paths:
        return config.get_path("paths", group, value)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (config.project_root / candidate).resolve()
    return candidate


def resolve_video_and_model_paths(args, config):
    video_path = resolve_config_or_project_path(config, "data", args.video_path)
    model_path = resolve_config_or_project_path(config, "models", args.model_path)
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el vídeo: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    return video_path, model_path


def slugify_model_name(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(value).stem) or "model"


def build_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
