"""Flujo offline completo: tracks JSON → adapter → PathCRF inference → events.

Comparte adapter, inference y postprocess con el flujo rolling/live.
"""

from .adapter import (
    PathCRFAdapterConfig,
    PathCRFTracksAdapter,
    convert_tracks_dict_to_pathcrf,
    convert_tracks_json_to_pathcrf,
)
from .inference import (
    PathCRFInferenceConfig,
    PathCRFRenderConfig,
    PathCRFPipelineResult,
    run_pathcrf_inference,
    run_pathcrf_pipeline,
)

__all__ = [
    "PathCRFAdapterConfig",
    "PathCRFInferenceConfig",
    "PathCRFPipelineResult",
    "PathCRFRenderConfig",
    "PathCRFTracksAdapter",
    "convert_tracks_dict_to_pathcrf",
    "convert_tracks_json_to_pathcrf",
    "run_pathcrf_inference",
    "run_pathcrf_pipeline",
]
