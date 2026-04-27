"""
Core module - Configuration, logging, and shared utilities.
"""

from .config import Config, get_config
from .logger import Logger, get_logger
from .phase_packets import (
    IDENTITY_HOMOGRAPHY_3X3,
    PHASE_BYTETRACK,
    PHASE_CANONICALTRACK,
    PHASE_DETECTOR,
    PHASE_FILTERING,
    PHASE_IDENTIFICATION,
    PHASE_POSESSION,
    PHASE_REFERENCE_POINTS,
    REJECT_CODE_HOMOGRAPHY_NOT_USABLE,
    REJECT_CODE_INVALID_FIELD_POSITION,
    REJECT_CODE_KEPT,
    REJECT_CODE_LABELS,
    REJECT_CODE_OUTSIDE_FIELD,
    REJECT_CODE_RESCUED_BY_TRACK_OVERLAP,
    default_render_color_bgr,
    make_phase_packet,
)
from .serialization import convert_to_serializable

__all__ = [
    "Config",
    "Logger",
    "IDENTITY_HOMOGRAPHY_3X3",
    "PHASE_BYTETRACK",
    "PHASE_CANONICALTRACK",
    "PHASE_DETECTOR",
    "PHASE_FILTERING",
    "PHASE_IDENTIFICATION",
    "PHASE_POSESSION",
    "PHASE_REFERENCE_POINTS",
    "REJECT_CODE_HOMOGRAPHY_NOT_USABLE",
    "REJECT_CODE_INVALID_FIELD_POSITION",
    "REJECT_CODE_KEPT",
    "REJECT_CODE_LABELS",
    "REJECT_CODE_OUTSIDE_FIELD",
    "REJECT_CODE_RESCUED_BY_TRACK_OVERLAP",
    "SCHEMA_VERSION",
    "convert_to_serializable",
    "default_render_color_bgr",
    "get_config",
    "get_logger",
    "make_phase_packet",
]
