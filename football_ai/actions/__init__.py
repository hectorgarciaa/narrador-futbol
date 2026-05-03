"""Actions module — rolling (live) and offline PathCRF inference."""

from .rolling import RollingActionsConfig, RollingActionsPhase
from .postprocess import (
    EdgeGroupStats,
    postprocess_emit_block,
    reset_postprocess_state,
)
from .inference import (
    PathCRFInferenceConfig,
    PathCRFPipelineResult,
    PathCRFRenderConfig,
    run_pathcrf_inference,
    run_pathcrf_pipeline,
)
from .model import get_or_load_model
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
    "ConversionSummary",
    "EdgeGroupStats",
    "PathCRFAdapterConfig",
    "PathCRFInferenceConfig",
    "PathCRFPipelineResult",
    "PathCRFRenderConfig",
    "PathCRFTracksAdapter",
    "RollingActionsConfig",
    "RollingActionsPhase",
    "apply_shot_heuristic",
    "build_commentary_events_json",
    "classify_episode_starts",
    "classify_setpieces",
    "convert_tracks_dict_to_pathcrf",
    "convert_tracks_json_to_pathcrf",
    "get_or_load_model",
    "postprocess_emit_block",
    "reset_postprocess_state",
    "run_pathcrf_inference",
    "run_pathcrf_pipeline",
]
