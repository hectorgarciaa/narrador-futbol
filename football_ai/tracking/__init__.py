"""
Tracking module — orquestación de las 6 subfases de tracking.

Exports:
  Tracker        — clase pura que encadena las 6 subfases.
  TrackingPhase  — wrapper Phase sobre Tracker para el pipeline modular.
  PHASE_TRACKING — constante de nombre de fase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .phase import TrackingPhase
    from .tracker import PHASE_TRACKING, Tracker


def __getattr__(name: str):
    if name == "TrackingPhase":
        from .phase import TrackingPhase as exported

        return exported
    if name in {"Tracker", "PHASE_TRACKING"}:
        from .tracker import PHASE_TRACKING, Tracker

        return {"Tracker": Tracker, "PHASE_TRACKING": PHASE_TRACKING}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Tracker", "TrackingPhase", "PHASE_TRACKING"]
