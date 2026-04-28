from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar `python scripts/actions/run_pathcrf.py` sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions import (  # noqa: E402
    PathCRFAdapterConfig,
    PathCRFInferenceConfig,
    PathCRFRenderConfig,
    run_pathcrf_pipeline,
)
from football_ai.core.config import Config  # noqa: E402
from football_ai.positions.data import resolve_tracks_path_for_video  # noqa: E402
from football_ai.pipeline.paths import resolve_video_path, sanitize_video_stem  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convierte la salida de `scripts/track.py` al formato de PathCRF, "
            "ejecuta inferencia y opcionalmente renderiza un video 2D con la arista activa."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Puede ser un shortcut de video de config.yaml (ej. video_prueba), "
            "una ruta a `<video>_tracks.json`, una ruta a un vídeo o un parquet ya convertido."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta donde guardar los artefactos de PathCRF. Por defecto: output/actions/pathcrf/<video>/",
    )
    parser.add_argument(
        "--tracking-output-path",
        type=Path,
        default=None,
        help="Ruta explícita para el parquet convertido desde tracks.json.",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path("football_ai/actions/repo/pathcrf"),
        help="Ruta al repo clonado de PathCRF.",
    )
    parser.add_argument("--trial", type=int, default=120, help="Trial/checkpoint de PathCRF a usar.")
    parser.add_argument(
        "--model-file",
        type=str,
        default="state_dict_best_acc.pt",
        help="Nombre del checkpoint dentro de `saved/<trial>/model` o ruta absoluta al checkpoint.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override de FPS para PathCRF.")
    parser.add_argument(
        "--sample-freq",
        type=int,
        default=None,
        help="Override del sample frequency de PathCRF.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Override de la ventana temporal usada por PathCRF.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo para inferencia (`auto`, `cpu`, `cuda:0`, ...).",
    )
    parser.add_argument(
        "--decode",
        choices=("indep", "greedy", "viterbi"),
        default="indep",
        help="Modo de decodificación cuando `--no-crf` está activo.",
    )
    parser.add_argument(
        "--no-crf",
        action="store_true",
        help="Desactiva el CRF del checkpoint y usa solo la salida micro.",
    )
    parser.add_argument(
        "--correct-episode-lasts",
        action="store_true",
        help="Activa la corrección del último nodo por episodio si el checkpoint lo soporta.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Activa métricas internas de PathCRF. En inferencia sobre clips sin GT suele dejarse apagado.",
    )
    parser.add_argument(
        "--min-event-duration",
        type=int,
        default=10,
        help="Duración mínima para compactar self-loops cortos al detectar eventos.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="No genera el video 2D del campo con la arista activa.",
    )
    parser.add_argument(
        "--render-output-path",
        type=Path,
        default=None,
        help="Ruta explícita del video MP4 de visualización PathCRF.",
    )
    parser.add_argument("--render-width", type=int, default=1280, help="Ancho del video renderizado.")
    parser.add_argument("--render-height", type=int, default=720, help="Alto del video renderizado.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la visualización en tiempo real si hay entorno gráfico disponible.",
    )
    return parser


def _infer_video_for_tracks(tracks_path: Path) -> Path | None:
    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    target_stem = sanitize_video_stem(tracks_path.stem.replace("_tracks", ""))
    for _, value in config.paths.get("data", {}).items():
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (config.project_root / candidate).resolve()
        if not candidate.exists() or candidate.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if sanitize_video_stem(candidate.stem) == target_stem:
            return candidate.resolve()
    return None


def resolve_source(source: str | None) -> tuple[Path | None, Path | None, Path | None]:
    if source is None:
        config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
        video_path, _ = resolve_video_path(config, None)
        tracks_path = resolve_tracks_path_for_video(PROJECT_ROOT, Path(video_path))
        return tracks_path.resolve(), None, Path(video_path).resolve()

    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    if candidate.exists():
        suffix = candidate.suffix.lower()
        if suffix == ".json":
            return candidate, None, _infer_video_for_tracks(candidate)
        if suffix == ".parquet":
            return None, candidate, None
        if suffix in VIDEO_SUFFIXES:
            tracks_path = resolve_tracks_path_for_video(PROJECT_ROOT, candidate)
            return tracks_path.resolve(), None, candidate.resolve()

    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    video_path, _ = resolve_video_path(config, source)
    tracks_path = resolve_tracks_path_for_video(PROJECT_ROOT, Path(video_path))
    return tracks_path.resolve(), None, Path(video_path).resolve()


def default_output_dir(tracks_path: Path | None, tracking_path: Path | None) -> Path:
    source_path = tracking_path or tracks_path
    if source_path is None:
        return PROJECT_ROOT / "output" / "actions" / "pathcrf" / "default"
    stem = sanitize_video_stem(source_path.stem)
    for suffix in ("_tracks", "_tracking"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return PROJECT_ROOT / "output" / "actions" / "pathcrf" / stem


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tracks_path, tracking_path, video_path = resolve_source(args.source)
    if tracks_path is not None and not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")
    if tracking_path is not None and not tracking_path.exists():
        raise FileNotFoundError(f"No existe el parquet indicado: {tracking_path}")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_output_dir(tracks_path, tracking_path)
    adapter_config = PathCRFAdapterConfig(fps=float(args.fps) if args.fps is not None else 25.0)
    inference_config = PathCRFInferenceConfig(
        repo_path=args.repo_path,
        trial=int(args.trial),
        model_file=str(args.model_file),
        use_crf=not bool(args.no_crf),
        decode=str(args.decode),
        correct_episode_lasts=bool(args.correct_episode_lasts),
        evaluate=bool(args.evaluate),
        window_seconds=float(args.window_seconds) if args.window_seconds is not None else None,
        fps=float(args.fps) if args.fps is not None else None,
        sample_freq=int(args.sample_freq) if args.sample_freq is not None else None,
        min_event_duration=int(args.min_event_duration),
        device=str(args.device),
    )
    render_config = PathCRFRenderConfig(
        enabled=not bool(args.no_render),
        width=int(args.render_width),
        height=int(args.render_height),
        show_window=bool(args.show),
    )

    result = run_pathcrf_pipeline(
        output_dir=output_dir,
        tracks_path=tracks_path,
        tracking_path=tracking_path,
        video_path=video_path,
        tracking_output_path=args.tracking_output_path.resolve() if args.tracking_output_path is not None else None,
        adapter_config=adapter_config,
        inference_config=inference_config,
        render_config=render_config,
        render_output_path=args.render_output_path.resolve() if args.render_output_path is not None else None,
    )

    print(f"Output dir: {output_dir}")
    print(f"Tracking parquet: {result.tracking_path}")
    if result.tracking_summary_path is not None:
        print(f"Tracking summary: {result.tracking_summary_path}")
    print(f"Edge sequence: {result.edge_sequence_path}")
    print(f"Events: {result.events_path}")
    if result.semantic_events_path is not None:
        print(f"Semantic events: {result.semantic_events_path}")
    if result.commentary_json_path is not None:
        print(f"Commentary JSON: {result.commentary_json_path}")
    if result.macro_prev_path is not None:
        print(f"Macro prev: {result.macro_prev_path}")
    if result.macro_next_path is not None:
        print(f"Macro next: {result.macro_next_path}")
    if result.render_path is not None:
        print(f"Render video: {result.render_path}")
    print(f"Summary JSON: {result.summary_path}")


if __name__ == "__main__":
    main()
