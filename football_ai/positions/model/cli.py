from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import TrainingConfig
from .inference import predict_roles_for_video
from .render import render_role_video
from .training import train_position_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_position_model(
            project_root=args.project_root,
            base_table_path=args.base_table_path,
            output_dir=args.output_dir,
            config=_training_config_from_args(args),
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            prefer_cached_common_dataset=not bool(args.rebuild_from_base_table),
        )
        print(json.dumps({
            "checkpoint_path": str(result["checkpoint_path"]),
            "output_dir": str(result["artifacts"].output_dir),
            "best_epoch": int(result["metrics"]["best_epoch"]),
            "val_macro_f1": float(result["metrics"]["val"]["macro_f1"]),
            "test_macro_f1": float(result["metrics"]["test"]["macro_f1"]),
            "trained_labels": result["metrics"]["trained_labels"],
            "missing_known_labels": result["metrics"]["missing_known_labels"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "predict":
        result = predict_roles_for_video(
            model_path=args.model_path,
            video_path=args.video_path,
            project_root=args.project_root,
            output_dir=args.output_dir,
            tracks_path=args.tracks_path,
            batch_size=int(args.batch_size),
        )
        print(json.dumps({
            "output_dir": str(result["output_dir"]),
            "frame_predictions_path": str(result["frame_predictions_path"]),
            "player_predictions_path": str(result["player_predictions_path"]),
            "tracks_with_roles_path": str(result["tracks_with_roles_path"]),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "render-video":
        result = render_role_video(
            video_path=args.video_path,
            tracks_path=args.tracks_path,
            project_root=args.project_root,
            output_path=args.output_path,
            show=bool(args.show),
        )
        print(json.dumps({
            "video_path": str(result["video_path"]),
            "tracks_path": str(result["tracks_path"]),
            "source_video_path": str(result["source_video_path"]),
        }, ensure_ascii=False, indent=2))
        return 0
    parser.error("Comando no soportado.")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrenamiento e inferencia de un Set Transformer para roles posicionales.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Entrenar modelo desde base_table.csv.")
    train_parser.add_argument("--project-root", type=Path, default=None)
    train_parser.add_argument("--base-table-path", type=Path, default=None)
    train_parser.add_argument("--output-dir", type=Path, default=None)
    train_parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train_parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    train_parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    train_parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    train_parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    train_parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-val-samples", type=int, default=None)
    train_parser.add_argument("--max-test-samples", type=int, default=None)
    train_parser.add_argument("--rebuild-from-base-table", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Aplicar checkpoint a un vídeo.")
    predict_parser.add_argument("--project-root", type=Path, default=None)
    predict_parser.add_argument("--model-path", type=Path, required=True)
    predict_parser.add_argument("--video-path", type=Path, required=True)
    predict_parser.add_argument("--tracks-path", type=Path, default=None)
    predict_parser.add_argument("--output-dir", type=Path, default=None)
    predict_parser.add_argument("--batch-size", type=int, default=1024)

    render_parser = subparsers.add_parser("render-video", help="Renderizar un MP4 anotado usando tracks JSON con roles predichos.")
    render_parser.add_argument("--project-root", type=Path, default=None)
    render_parser.add_argument("--video-path", type=Path, required=True)
    render_parser.add_argument("--tracks-path", type=Path, required=True)
    render_parser.add_argument("--output-path", type=Path, default=None)
    render_parser.add_argument("--show", action="store_true")
    return parser


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        patience=int(args.patience),
    )
