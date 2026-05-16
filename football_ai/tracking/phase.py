"""
TrackingPhase — wrapper Phase sobre Tracker.

Permite tratar las 6 subfases de tracking como una única fase más
del pipeline, consumible igual que PosessionPhase o PositionInferingPhase.

    phase = TrackingPhase(...)
    phase.reset()
    packet, elapsed_ms = phase.process(frame_bgr, frame_index, frame_time_ms)
"""
from __future__ import annotations

from football_ai.core import Phase

from .tracker import Tracker


class TrackingPhase(Phase):
    """
    Fase de tracking unificada.

    Hereda de Phase: expone `execute` (llamado por `process`) y delega
    toda la lógica real a `Tracker.process_frame`.
    """

    def __init__(self, tracking_runtime_conf):
        self.tracker = Tracker(tracking_runtime_conf)

    def reset(self) -> None:
        """Delega el reset al Tracker interno."""
        self.tracker.reset()

    def execute(
        self,
        frame_bgr,
        frame_index: int,
        frame_time_ms: float,
        *,
        show_kmeans: bool = False,
        collect_visual_debug: bool = False,
        execution_mode: str | None = None,
    ) -> dict:
        """
        Ejecuta el pipeline de tracking completo y devuelve el packet
        canónico listo para las fases siguientes (Posession, Positions…).
        """
        return self.tracker.process_frame(
            frame_bgr,
            frame_index,
            frame_time_ms,
            show_kmeans=show_kmeans,
            collect_visual_debug=collect_visual_debug,
            execution_mode=execution_mode,
        )


__all__ = ["TrackingPhase"]
