from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions import (  # noqa: E402
    LiveSnapshotsConfig,
    PathCRFAdapterConfig,
    PathCRFInferenceConfig,
    run_live_snapshots_pipeline,
)
from football_ai.core.config import Config  # noqa: E402
from football_ai.pipeline.paths import resolve_video_path  # noqa: E402
from football_ai.positions.data import resolve_tracks_path_for_video  # noqa: E402


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
            "Pipeline live-snapshots aislado: reejecuta PathCRF offline N veces "
            "sobre snapshots acumulados del tracks.json."
        )
    )
    parser.add_argument("source", nargs="?", default=None, help="Video key/ruta o tracks.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Salida. Default: output/actions/pathcrf_live_snapshots/<video>/")
    parser.add_argument("--repo-path", type=Path, default=Path("football_ai/actions/repo/pathcrf"))
    parser.add_argument("--trial", type=int, default=120)
    parser.add_argument("--model-file", type=str, default="state_dict_best_acc.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--snapshot-interval-frames", type=int, default=75)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--decode", choices=("indep", "greedy", "viterbi"), default="indep")
    parser.add_argument("--no-crf", action="store_true")
    parser.add_argument("--correct-episode-lasts", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--sample-freq", type=int, default=None)
    parser.add_argument("--min-event-duration", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tracks_path = _resolve_tracks(args.source)
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / "output" / "actions" / "pathcrf_live_snapshots" / tracks_path.stem.replace("_tracks", "")).resolve()
    )
    result = run_live_snapshots_pipeline(
        tracks_path=tracks_path,
        output_dir=out_dir,
        config=LiveSnapshotsConfig(
            snapshot_interval_frames=int(args.snapshot_interval_frames),
            min_frames=int(args.min_frames),
            fps=float(args.fps),
            adapter_config=PathCRFAdapterConfig(fps=float(args.fps)),
            inference_config=PathCRFInferenceConfig(
                repo_path=args.repo_path,
                trial=int(args.trial),
                model_file=str(args.model_file),
                use_crf=not bool(args.no_crf),
                decode=str(args.decode),
                correct_episode_lasts=bool(args.correct_episode_lasts),
                evaluate=bool(args.evaluate),
                window_seconds=float(args.window_seconds) if args.window_seconds is not None else None,
                fps=float(args.fps),
                sample_freq=int(args.sample_freq) if args.sample_freq is not None else None,
                min_event_duration=int(args.min_event_duration),
                device=str(args.device),
            ),
        ),
    )
    print(f"Output dir: {result.output_dir}")
    print(f"Frames analizados: {result.frame_count}")
    print(f"Snapshot runs: {len(result.snapshot_runs)}")
    print(f"Sparse edge sequence: {result.sparse_edge_sequence_path}")
    print(f"Summary JSON: {result.summary_path}")


if __name__ == "__main__":
    main()

