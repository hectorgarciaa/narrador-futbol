"""
Identification module - Shirt and team detection.
"""

from .shirt_detector import ShirtDetector
from .phase import IdentificationPhase
from .team_detector import TeamDetector


__all__ = ["ShirtDetector", "TeamDetector", "IdentificationPhase"]
