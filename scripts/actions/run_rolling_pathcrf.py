from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions_realtime import (
    RollingRuntimeConfig,
    run_rolling_pipeline,
)
from football_ai.core.config import Config
from football_ai.pipeline.paths import resolve_video_path
from football_ai.positions.data import resolve_tracks_path_for_video


def _resolve_tracks(source: str | None) -> Path:
    if source is None:
        config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
        video_path, _ = resolve_video_path(config, None)
        return resolve_tracks_path_for_video(PROJECT_ROOT, Path(video_path)).resolve()
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if candidate.exists() and candidate.suffix.lower() == ".json":
        return candidate
    if candidate.exists():
        return resolve_tracks_path_for_video(PROJECT_ROOT, candidate).resolve()
    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    video_path, _ = resolve_video_path(config, source)
    return resolve_tracks_path_for_video(PROJECT_ROOT, Path(video_path)).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline rolling PathCRF: modo default basado en snapshots (adapter offline) "
            "para emular el bridge live. --tracking-path queda como modo diagnostico."
        )
    )
    parser.add_argument("source", nargs="?", default=None, help="Video key/ruta o tracks.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Salida. Default: output/actions/pathcrf_rolling/<video>/",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path("football_ai/actions/repo/pathcrf"),
    )
    parser.add_argument("--trial", type=int, default=120)
    parser.add_argument("--model-file", type=str, default="state_dict_best_acc.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fps", type=float, default=25.0)

    parser.add_argument("--cadence-frames", type=int, default=10)
    parser.add_argument("--context-frames", type=int, default=250)
    parser.add_argument("--emit-delay-frames", type=int, default=25)
    parser.add_argument("--emit-frames", type=int, default=10)
    parser.add_argument("--min-frames-warmup", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--decode", choices=("indep", "greedy", "viterbi"), default="indep")
    parser.add_argument("--no-crf", action="store_true")
    parser.add_argument("--correct-episode-lasts", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--sample-freq", type=int, default=5)
    parser.add_argument("--min-event-duration", type=int, default=10)

    parser.add_argument(
        "--tracking-path",
        type=Path,
        default=None,
        help="Usar un tracking.parquet pre-construido (modo diagnostico/upper-bound).",
    )
    parser.add_argument("--smooth-edges", action="store_true")
    parser.add_argument("--edge-smooth-window", type=int, default=3)
    parser.add_argument(
        "--tracking-diff",
        action="store_true",
        help="Comparar tracking por snapshot contra el offline completo y guardar diffs.",
    )
    parser.add_argument(
        "--tracking-diff-atol",
        type=float,
        default=1e-6,
        help="Tolerancia absoluta para comparar columnas numericas en diffs de tracking.",
    )
    parser.add_argument(
        "--tracking-diff-max-rows",
        type=int,
        default=5000,
        help="Maximo de filas por diff de tracking antes de truncar.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    tracks_path = _resolve_tracks(args.source)
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            PROJECT_ROOT
            / "output"
            / "actions"
            / "pathcrf_rolling"
            / tracks_path.stem.replace("_tracks", "")
        ).resolve()
    )

    config = RollingRuntimeConfig(
        cadence_frames=int(args.cadence_frames),
        context_frames=int(args.context_frames),
        emit_delay_frames=int(args.emit_delay_frames),
        emit_frames=int(args.emit_frames),
        fps=float(args.fps),
        sample_freq=int(args.sample_freq),
        window_seconds=float(args.window_seconds),
        use_crf=not bool(args.no_crf),
        decode=str(args.decode),
        min_event_duration=int(args.min_event_duration),
        min_frames_warmup=int(args.min_frames_warmup),
        repo_path=args.repo_path,
        trial=int(args.trial),
        model_file=str(args.model_file),
        device=str(args.device),
        correct_episode_lasts=bool(args.correct_episode_lasts),
        evaluate=bool(args.evaluate),
        smooth_edges=bool(args.smooth_edges),
        edge_smooth_window=int(args.edge_smooth_window),
        tracking_diff_enabled=bool(args.tracking_diff),
        tracking_diff_atol=float(args.tracking_diff_atol),
        tracking_diff_max_rows=int(args.tracking_diff_max_rows),
    )

    result = run_rolling_pipeline(
        tracks_path=tracks_path,
        output_dir=out_dir,
        config=config,
        max_frames=int(args.max_frames) if args.max_frames is not None else None,
        tracking_path=args.tracking_path.resolve() if args.tracking_path is not None else None,
    )

    print(f"\nOutput dir: {out_dir}")
    print(f"Tracking: {result.tracking_path}")
    print(f"Rolling edges: {result.rolling_edges_path}")
    print(f"Emitted edges: {result.emitted_edges_path}")
    print(f"Checkpoints: {result.checkpoints_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Frames: {result.frame_count}")
    print(f"Checkpoints: {result.num_checkpoints}")
    print(f"Total emitted edges: {result.total_emitted_edges}")


if __name__ == "__main__":
    main()