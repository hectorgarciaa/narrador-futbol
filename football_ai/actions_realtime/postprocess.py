from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RealtimeEdge:
    frame_id: int
    edge_src: str | None
    edge_dst: str | None
    canonical_src: str | None
    canonical_dst: str | None


@dataclass
class RealtimeAction:
    frame_id: int
    edge_src: str | None
    edge_dst: str | None
    canonical_src: str | None
    canonical_dst: str | None
    event_type: str
    player_id: str | None
    receiver_id: str | None
    timestamp: str | None
    start_x: float | None
    start_y: float | None
    end_x: float | None
    end_y: float | None
    confidence: float | None
    is_realtime: bool = True
    is_final: bool = False


@dataclass
class RealtimeCheckpoint:
    checkpoint_frame: int
    snapshot_frames: int
    edge_rows: int
    emit_start: int
    emit_end: int
    emitted_count: int
    infer_ms: float
    total_emitted_so_far: int


_SELF_EDGE_EVENT = "control"
_PASS_EVENT = "pase"
_OUT_EVENT = "out"
_UNKNOWN_EVENT = "unknown"


def _is_self(pair: tuple[str | None, str | None]) -> bool:
    src, dst = pair
    if src is None or dst is None:
        return False
    return src == dst


def _is_out(slot_name: str | None) -> bool:
    if slot_name is None:
        return False
    lower = str(slot_name).strip().lower()
    return bool(lower in ("out", "outside", "_out_") or lower.startswith("out"))


def _classify_event(src: str | None, dst: str | None) -> str:
    if src is None or dst is None:
        return _UNKNOWN_EVENT
    if _is_self((src, dst)):
        return _SELF_EDGE_EVENT
    if _is_out(src) or _is_out(dst):
        return _OUT_EVENT
    return _PASS_EVENT


def edge_to_action(
    edge: dict[str, Any] | RealtimeEdge,
    tracker_state: dict[str, dict[str, Any]] | None = None,
) -> RealtimeAction:
    frame_id = int(edge.get("frame_id", 0))
    src = edge.get("edge_src")
    dst = edge.get("edge_dst")
    canonical_src = edge.get("canonical_src")
    canonical_dst = edge.get("canonical_dst")
    event_type = _classify_event(src, dst)

    player_id = canonical_src
    receiver_id = canonical_dst if not _is_self((src, dst)) else None

    start_x = None
    start_y = None
    end_x = None
    end_y = None

    if tracker_state:
        src_slot = str(src) if src else None
        dst_slot = str(dst) if dst else None
        if src_slot in tracker_state:
            pos = tracker_state[src_slot]
            start_x = round(float(pos.get("x", None)), 2) if pos.get("x") is not None else None
            start_y = round(float(pos.get("y", None)), 2) if pos.get("y") is not None else None
        if dst_slot in tracker_state:
            pos = tracker_state[dst_slot]
            end_x = round(float(pos.get("x", None)), 2) if pos.get("x") is not None else None
            end_y = round(float(pos.get("y", None)), 2) if pos.get("y") is not None else None

    return RealtimeAction(
        frame_id=frame_id,
        edge_src=src,
        edge_dst=dst,
        canonical_src=canonical_src,
        canonical_dst=canonical_dst,
        event_type=event_type,
        player_id=player_id,
        receiver_id=receiver_id,
        timestamp=str(frame_id / 25.0) if 25.0 else None,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        confidence=None,
        is_realtime=True,
        is_final=True,
    )


def postprocess_emitted_edges(
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [{**a.__dict__} for a in (edge_to_action(e) for e in edges)]
