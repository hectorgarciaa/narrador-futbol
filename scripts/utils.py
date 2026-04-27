import argparse
import re
import cv2
from datetime import datetime
from pathlib import Path

from football_ai.core import get_config
from football_ai.detection import Detector


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

def prepare_context(args, config, output_sub_dir):
    video_path, model_path = resolve_video_and_model_paths(args, config)

    timestamp = build_timestamp()
    output_dir = PROJECT_ROOT / "output" / output_sub_dir / slugify_model_name(model_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return fps, width, height, total_frames, output_dir, video_path, model_path


def iter_video_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval_ms = 1000.0 / float(fps if fps > 0 else 25.0)
    frame_index = 0
    while True:
        if max_frames is not None and frame_index >= int(max_frames):
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        yield frame_index, frame_index * frame_interval_ms, frame_bgr
        frame_index += 1
    cap.release()
