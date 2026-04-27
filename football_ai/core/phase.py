from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any


class Phase(ABC):
    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> dict:
        """Run the phase logic and return a phase packet."""

    def process(self, *args: Any, **kwargs: Any) -> tuple[dict, float]:
        start = perf_counter()
        packet = self.execute(*args, **kwargs)
        elapsed_ms = (perf_counter() - start) * 1000.0
        if isinstance(packet, dict):
            packet["elapsed_ms"] = float(elapsed_ms)
        return packet, float(elapsed_ms)
