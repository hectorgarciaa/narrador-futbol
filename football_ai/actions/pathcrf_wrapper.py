from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from football_ai.core.serialization import convert_to_serializable
from football_ai.tracking_cli.paths import sanitize_video_stem
from football_ai.visualization.pathcrf_drawer import PathCRFDrawer

from .pathcrf_adapter import ConversionSummary, PathCRFAdapterConfig, convert_tracks_json_to_pathcrf


@dataclass(frozen=True)
class PathCRFInferenceConfig:
    repo_path: Path = Path("football_ai/actions/repo/pathcrf")
    trial: int = 120
    model_file: str = "state_dict_best_acc.pt"
    use_crf: bool = True
    decode: str = "indep"
    correct_episode_lasts: bool = False
    evaluate: bool = False
    window_seconds: float | None = None
    fps: float | None = None
    sample_freq: int | None = None
    min_event_duration: int = 10
    device: str = "auto"


@dataclass(frozen=True)
class PathCRFRenderConfig:
    enabled: bool = True
    width: int = 1280
    height: int = 720
    show_window: bool = False


@dataclass
class PathCRFPipelineResult:
    tracks_path: Path | None
    tracking_path: Path
    tracking_summary_path: Path | None
    edge_sequence_path: Path
    events_path: Path
    macro_prev_path: Path | None
    macro_next_path: Path | None
    render_path: Path | None
    summary_path: Path
    stats: dict[str, Any]
    conversion_summary: ConversionSummary | None = None


def _resolve_repo_path(repo_path: str | Path) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el repo de PathCRF: {path}")
    return path


def _ensure_pathcrf_import_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _install_torch_geometric_stub() -> None:
    if "torch_geometric" in sys.modules:
        return

    tg_module = types.ModuleType("torch_geometric")
    tg_data_module = types.ModuleType("torch_geometric.data")

    class _DummyData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _DummyBatch:
        @staticmethod
        def from_data_list(data_list: list[Any]) -> list[Any]:
            return data_list

    tg_data_module.Data = _DummyData
    tg_data_module.Batch = _DummyBatch
    tg_module.data = tg_data_module
    sys.modules["torch_geometric"] = tg_module
    sys.modules["torch_geometric.data"] = tg_data_module


def _has_real_torch_geometric() -> bool:
    try:
        importlib.import_module("torch_geometric.data")
        return True
    except ModuleNotFoundError:
        return False


def _select_device(device_name: str) -> torch.device:
    normalized = str(device_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _import_pathcrf_modules(repo_path: Path, trial_args: dict[str, Any]) -> tuple[Any, Any, Any]:
    _ensure_pathcrf_import_path(repo_path)
    agent_model = str(trial_args.get("agent_model", "")).strip().lower()
    if agent_model != "gat" and not _has_real_torch_geometric():
        _install_torch_geometric_stub()
    elif agent_model == "gat" and not _has_real_torch_geometric():
        raise ModuleNotFoundError(
            "El checkpoint de PathCRF usa `agent_model=gat`, pero `torch_geometric` no está instalado en el entorno."
        )

    pathcrf_utils = importlib.import_module("models.utils")
    pathcrf_inference = importlib.import_module("inference")
    pathcrf_postprocess = importlib.import_module("datatools.postprocess")
    return pathcrf_utils, pathcrf_inference, pathcrf_postprocess


def _normalize_frame_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "frame_id" in result.columns:
        return result
    frame_index_name = result.index.name or "frame_id"
    result = result.reset_index()
    if frame_index_name != "frame_id" and frame_index_name in result.columns:
        result = result.rename(columns={frame_index_name: "frame_id"})
    elif "index" in result.columns:
        result = result.rename(columns={"index": "frame_id"})
    return result


def _infer_output_stem(path: str | Path) -> str:
    stem = sanitize_video_stem(Path(path).stem)
    for suffix in (
        "_tracks",
        "_tracking",
        "_edge_sequence",
        "_events",
        "_macro_prev",
        "_macro_next",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or "pathcrf"


def _write_summary_json(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(payload), f, indent=2, ensure_ascii=False)


def _ensure_events_schema(events_df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = [
        "frame_id",
        "period_id",
        "episode_id",
        "timestamp",
        "player_id",
        "receiver_id",
        "event_type",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]
    result = events_df.copy()
    for column in expected_columns:
        if column not in result.columns:
            result[column] = pd.Series(dtype=object)
    return result[expected_columns]


def run_pathcrf_inference(
    tracking_path: str | Path,
    output_dir: str | Path,
    config: PathCRFInferenceConfig | None = None,
) -> PathCRFPipelineResult:
    inference_config = config or PathCRFInferenceConfig()
    tracking_path = Path(tracking_path).expanduser().resolve()
    if not tracking_path.exists():
        raise FileNotFoundError(f"No existe el parquet de tracking para PathCRF: {tracking_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = _infer_output_stem(tracking_path)

    repo_path = _resolve_repo_path(inference_config.repo_path)
    save_path = repo_path / "saved" / f"{int(inference_config.trial):03d}"
    if not save_path.exists():
        raise FileNotFoundError(f"No existe el trial de PathCRF: {save_path}")

    with (save_path / "args.json").open("r", encoding="utf-8") as f:
        trial_args = json.load(f)

    pathcrf_utils, pathcrf_inference, pathcrf_postprocess = _import_pathcrf_modules(repo_path, trial_args)
    device = _select_device(inference_config.device)
    model = pathcrf_utils.build_model(trial_args, device=device)
    model_path = Path(pathcrf_utils.resolve_model_path(str(save_path), inference_config.model_file)).resolve()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    tracking_df = pd.read_parquet(tracking_path)
    effective_fps = (
        float(inference_config.fps)
        if inference_config.fps is not None
        else float(trial_args.get("fps", 25.0))
    )
    effective_sample_freq = (
        int(inference_config.sample_freq)
        if inference_config.sample_freq is not None
        else int(trial_args.get("sample_freq", 5))
    )
    effective_window_seconds = (
        float(inference_config.window_seconds)
        if inference_config.window_seconds is not None
        else trial_args.get("window_seconds")
    )

    macro_prev_df, macro_next_df, micro_pred_df, stats = pathcrf_inference.inference(
        model=model,
        tracking=tracking_df,
        use_crf=bool(inference_config.use_crf),
        decode=str(inference_config.decode),
        correct_episode_lasts=bool(inference_config.correct_episode_lasts),
        evaluate=bool(inference_config.evaluate),
        window_seconds=effective_window_seconds,
        fps=effective_fps,
        sample_freq=effective_sample_freq,
    )

    if {"edge_src", "edge_dst"}.issubset(micro_pred_df.columns):
        edge_sequence_df = micro_pred_df.copy()
    else:
        edge_sequence_df = pathcrf_postprocess.edge_probs_to_seq(micro_pred_df)

    events_df = pathcrf_postprocess.detect_events(
        tracking=tracking_df,
        edge_seq=edge_sequence_df,
        min_dur=int(inference_config.min_event_duration),
    )
    events_df = _ensure_events_schema(events_df)

    edge_sequence_path = output_dir / f"{output_stem}_edge_sequence.parquet"
    events_path = output_dir / f"{output_stem}_events.parquet"
    macro_prev_path = output_dir / f"{output_stem}_macro_prev.parquet" if macro_prev_df is not None else None
    macro_next_path = output_dir / f"{output_stem}_macro_next.parquet" if macro_next_df is not None else None
    summary_path = output_dir / f"{output_stem}_summary.json"

    _normalize_frame_dataframe(edge_sequence_df).to_parquet(edge_sequence_path, index=False)
    events_df.to_parquet(events_path, index=False)
    if macro_prev_df is not None and macro_prev_path is not None:
        _normalize_frame_dataframe(macro_prev_df).to_parquet(macro_prev_path, index=False)
    if macro_next_df is not None and macro_next_path is not None:
        _normalize_frame_dataframe(macro_next_df).to_parquet(macro_next_path, index=False)

    summary_payload = {
        "tracking_path": str(tracking_path),
        "repo_path": str(repo_path),
        "trial": int(inference_config.trial),
        "trial_args": trial_args,
        "model_path": str(model_path),
        "device": str(device),
        "use_crf": bool(inference_config.use_crf),
        "decode": str(inference_config.decode),
        "correct_episode_lasts": bool(inference_config.correct_episode_lasts),
        "evaluate": bool(inference_config.evaluate),
        "fps": effective_fps,
        "sample_freq": effective_sample_freq,
        "window_seconds": effective_window_seconds,
        "min_event_duration": int(inference_config.min_event_duration),
        "edge_sequence_path": str(edge_sequence_path),
        "events_path": str(events_path),
        "macro_prev_path": str(macro_prev_path) if macro_prev_path is not None else None,
        "macro_next_path": str(macro_next_path) if macro_next_path is not None else None,
        "edge_rows": int(len(edge_sequence_df)),
        "event_rows": int(len(events_df)),
        "stats": stats,
    }
    _write_summary_json(summary_path, summary_payload)

    return PathCRFPipelineResult(
        tracks_path=None,
        tracking_path=tracking_path,
        tracking_summary_path=None,
        edge_sequence_path=edge_sequence_path,
        events_path=events_path,
        macro_prev_path=macro_prev_path,
        macro_next_path=macro_next_path,
        render_path=None,
        summary_path=summary_path,
        stats=stats,
        conversion_summary=None,
    )


def run_pathcrf_pipeline(
    *,
    output_dir: str | Path,
    tracks_path: str | Path | None = None,
    tracking_path: str | Path | None = None,
    video_path: str | Path | None = None,
    tracking_output_path: str | Path | None = None,
    adapter_config: PathCRFAdapterConfig | None = None,
    inference_config: PathCRFInferenceConfig | None = None,
    render_config: PathCRFRenderConfig | None = None,
    render_output_path: str | Path | None = None,
) -> PathCRFPipelineResult:
    if tracks_path is None and tracking_path is None:
        raise ValueError("Hay que indicar `tracks_path` o `tracking_path` para ejecutar PathCRF.")

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_tracks_path = Path(tracks_path).expanduser().resolve() if tracks_path is not None else None
    resolved_tracking_path = Path(tracking_path).expanduser().resolve() if tracking_path is not None else None
    resolved_video_path = Path(video_path).expanduser().resolve() if video_path is not None else None
    conversion_summary = None
    tracking_summary_path = None

    if resolved_tracking_path is None:
        if resolved_tracks_path is None or not resolved_tracks_path.exists():
            raise FileNotFoundError(f"No existe el tracks JSON indicado: {resolved_tracks_path}")
        output_stem = _infer_output_stem(resolved_tracks_path)
        resolved_tracking_path = (
            Path(tracking_output_path).expanduser().resolve()
            if tracking_output_path is not None
            else output_dir / f"{output_stem}_tracking.parquet"
        )
        resolved_tracking_path, conversion_summary = convert_tracks_json_to_pathcrf(
            tracks_path=resolved_tracks_path,
            output_path=resolved_tracking_path,
            config=adapter_config,
        )
        tracking_summary_path = resolved_tracking_path.with_suffix(".summary.json")
    else:
        if not resolved_tracking_path.exists():
            raise FileNotFoundError(f"No existe el parquet indicado: {resolved_tracking_path}")

    inference_result = run_pathcrf_inference(
        tracking_path=resolved_tracking_path,
        output_dir=output_dir,
        config=inference_config,
    )

    render_cfg = render_config or PathCRFRenderConfig()
    render_path = None
    if render_cfg.enabled:
        output_stem = _infer_output_stem(resolved_tracking_path)
        render_path = (
            Path(render_output_path).expanduser().resolve()
            if render_output_path is not None
            else output_dir / f"{output_stem}_pitch_pathcrf.mp4"
        )
        drawer = PathCRFDrawer()
        with inference_result.summary_path.open("r", encoding="utf-8") as f:
            rendered_summary = json.load(f)
        effective_render_fps = float(rendered_summary.get("fps", 25.0))

        drawer.render_tracking_and_edges(
            tracking=pd.read_parquet(resolved_tracking_path),
            edge_sequence=pd.read_parquet(inference_result.edge_sequence_path),
            output_path=render_path,
            events=pd.read_parquet(inference_result.events_path),
            fps=effective_render_fps,
            frame_size=(int(render_cfg.width), int(render_cfg.height)),
            video_path=resolved_video_path,
            tracks_path=resolved_tracks_path,
            conversion_summary=asdict(conversion_summary) if conversion_summary is not None else None,
            show=bool(render_cfg.show_window),
        )

    summary_payload: dict[str, Any]
    with inference_result.summary_path.open("r", encoding="utf-8") as f:
        summary_payload = json.load(f)
    summary_payload["tracks_path"] = str(resolved_tracks_path) if resolved_tracks_path is not None else None
    summary_payload["video_path"] = str(resolved_video_path) if resolved_video_path is not None else None
    summary_payload["tracking_path"] = str(resolved_tracking_path)
    summary_payload["tracking_summary_path"] = str(tracking_summary_path) if tracking_summary_path is not None else None
    summary_payload["render_path"] = str(render_path) if render_path is not None else None
    if conversion_summary is not None:
        summary_payload["conversion_summary"] = asdict(conversion_summary)
    _write_summary_json(inference_result.summary_path, summary_payload)

    return PathCRFPipelineResult(
        tracks_path=resolved_tracks_path,
        tracking_path=resolved_tracking_path,
        tracking_summary_path=tracking_summary_path,
        edge_sequence_path=inference_result.edge_sequence_path,
        events_path=inference_result.events_path,
        macro_prev_path=inference_result.macro_prev_path,
        macro_next_path=inference_result.macro_next_path,
        render_path=render_path,
        summary_path=inference_result.summary_path,
        stats=inference_result.stats,
        conversion_summary=conversion_summary,
    )
