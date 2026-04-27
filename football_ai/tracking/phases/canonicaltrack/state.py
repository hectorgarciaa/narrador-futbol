from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CanonicalTrackState:
    raw_to_canonical_id: dict[int, int] = field(default_factory=dict)
    canonical_to_raw_id: dict[int, int] = field(default_factory=dict)
    canonical_state: dict[int, dict] = field(default_factory=dict)
    forced_absorption_state: dict[int, dict] = field(default_factory=dict)
    ball_state: dict | None = None
    tracks_history: dict[str, list[dict]] = field(
        default_factory=lambda: {
            "player": [],
            "goalkeeper": [],
            "referee": [],
            "ball": [],
        }
    )
