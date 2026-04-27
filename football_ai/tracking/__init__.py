"""
Tracking module — orquestación de las 6 subfases de tracking.

Exports:
  Tracker        — clase pura que encadena las 6 subfases.
  TrackingPhase  — wrapper Phase sobre Tracker para el pipeline modular.
  PHASE_TRACKING — constante de nombre de fase.
"""

from .phase import TrackingPhase
from .tracker import PHASE_TRACKING, Tracker

__all__ = ["Tracker", "TrackingPhase", "PHASE_TRACKING"]
