"""Actions module — rolling (live) and offline PathCRF inference."""

from .rolling import RollingActionsConfig, RollingActionsPhase
from .postprocess import (
    EdgeGroupStats,
    RealtimeAction,
    RealtimeCheckpoint,
    RealtimeEdge,
    _classify_event,
    edge_to_action,
    postprocess_emit_block,
    postprocess_emitted_edges,
    reset_postprocess_state,
)
from .inference import (
    PathCRFInferenceConfig,
    PathCRFPipelineResult,
    PathCRFRenderConfig,
    run_pathcrf_inference,
    run_pathcrf_pipeline,
)
from .model import (
    _clear_model_cache,
    _select_device,
    get_or_load_model,
)
from .adapter import (
    ConversionSummary,
    PathCRFAdapterConfig,
    PathCRFTracksAdapter,
    convert_tracks_dict_to_pathcrf,
    convert_tracks_json_to_pathcrf,
)
from .pathcrf_commentary import build_commentary_events_json
from .pathcrf_setpieces import classify_episode_starts, classify_setpieces
from .pathcrf_shot import apply_shot_heuristic

__all__ = [
    "_classify_event",
    "_clear_model_cache",
    "_select_device",
    "ConversionSummary",
    "EdgeGroupStats",
    "PathCRFAdapterConfig",
    "PathCRFInferenceConfig",
    "PathCRFPipelineResult",
    "PathCRFRenderConfig",
    "PathCRFTracksAdapter",
    "RealtimeAction",
    "RealtimeCheckpoint",
    "RealtimeEdge",
    "RollingActionsConfig",
    "RollingActionsPhase",
    "apply_shot_heuristic",
    "build_commentary_events_json",
    "classify_episode_starts",
    "classify_setpieces",
    "convert_tracks_dict_to_pathcrf",
    "convert_tracks_json_to_pathcrf",
    "edge_to_action",
    "get_or_load_model",
    "postprocess_emit_block",
    "postprocess_emitted_edges",
    "reset_postprocess_state",
    "run_pathcrf_inference",
    "run_pathcrf_pipeline",
]
