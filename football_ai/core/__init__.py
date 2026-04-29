"""
Core module - Configuration, logging, and shared utilities.
"""

from .config import Config, get_config
from .logger import Logger, get_logger
from .phase import Phase
from .phase_packets import (
    IDENTITY_HOMOGRAPHY_3X3,
    PHASE_BYTETRACK,
    PHASE_CANONICALTRACK,
    PHASE_DETECTOR,
    PHASE_ACTIONS_DETECTOR,
    PHASE_FILTERING,
    PHASE_IDENTIFICATION,
    PHASE_POSESSION,
    PHASE_POSITION_INFERING,
    PHASE_PROJECTION,
    PHASE_REFERENCE_POINTS,
    default_render_color_bgr,
    make_phase_packet,
)
from .serialization import convert_to_serializable

__all__ = [
    "Config",
    "Logger",
    "Phase",
    "IDENTITY_HOMOGRAPHY_3X3",
    "PHASE_BYTETRACK",
    "PHASE_CANONICALTRACK",
    "PHASE_DETECTOR",
    "PHASE_ACTIONS_DETECTOR",
    "PHASE_FILTERING",
    "PHASE_IDENTIFICATION",
    "PHASE_POSESSION",
    "PHASE_POSITION_INFERING",
    "PHASE_PROJECTION",
    "PHASE_REFERENCE_POINTS",
    "SCHEMA_VERSION",
    "convert_to_serializable",
    "default_render_color_bgr",
    "get_config",
    "get_logger",
    "make_phase_packet",
]
