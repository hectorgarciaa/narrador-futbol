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
_PASS_EVENT = "kick"
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


# ── Temporal postprocessing: agrupar edges por bloque emitido ──

@dataclass
class EdgeGroupStats:
    canonical_src: str
    canonical_dst: str
    edge_src: str | None
    edge_dst: str | None
    support_frames: int
    longest_consecutive_run: int
    first_frame: int
    last_frame: int
    support_ratio: float

    @property
    def event_type(self) -> str:
        return _classify_event(self.edge_src, self.edge_dst)


def _compute_edge_groups(
    edges: list[dict[str, Any]],
    total_emit_frames: int,
) -> list[EdgeGroupStats]:
    """Agrupa edges por (canonical_src, canonical_dst) y calcula estadisticas temporales."""
    if not edges:
        return []

    groups: dict[tuple[str | None, str | None], list[int]] = {}
    for e in edges:
        cs = e.get("canonical_src") or e.get("edge_src")
        cd = e.get("canonical_dst") or e.get("edge_dst")
        if cs is None or cd is None:
            continue
        key = (str(cs), str(cd))
        groups.setdefault(key, []).append(int(e.get("frame_id", 0)))

    stats: list[EdgeGroupStats] = []
    for (cs, cd), frames in groups.items():
        frames_sorted = sorted(frames)
        if not frames_sorted:
            continue
        # longest consecutive run
        max_run = 1
        curr_run = 1
        for i in range(1, len(frames_sorted)):
            if frames_sorted[i] == frames_sorted[i - 1] + 1:
                curr_run += 1
                max_run = max(max_run, curr_run)
            else:
                curr_run = 1

        # find representative edge (pick one with actual src/dst strings)
        rep = next((e for e in edges
                     if str(e.get("canonical_src") or e.get("edge_src") or "") == cs
                     and str(e.get("canonical_dst") or e.get("edge_dst") or "") == cd), {})

        stats.append(EdgeGroupStats(
            canonical_src=cs,
            canonical_dst=cd,
            edge_src=rep.get("edge_src"),
            edge_dst=rep.get("edge_dst"),
            support_frames=len(frames_sorted),
            longest_consecutive_run=max_run,
            first_frame=frames_sorted[0],
            last_frame=frames_sorted[-1],
            support_ratio=len(frames_sorted) / max(1, total_emit_frames),
        ))

    return sorted(stats, key=lambda s: (-s.support_frames, -s.longest_consecutive_run))


_default_postprocess_config = {
    "min_support_frames": 5,
    "min_consecutive_frames": 3,
    "min_support_ratio": 0.15,
    "allow_multiple_events_per_emit": False,
    "deduplicate_events": True,
    "hysteresis_margin": 0.10,
}


# State for hysteresis between emit blocks
_last_emitted_key: tuple[str, str] | None = None
_last_emitted_frames: int = 0


def reset_postprocess_state() -> None:
    global _last_emitted_key, _last_emitted_frames
    _last_emitted_key = None
    _last_emitted_frames = 0


def postprocess_emit_block(
    edges: list[dict[str, Any]],
    total_emit_frames: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Postprocesa un bloque de edges emitidos y devuelve acciones consolidadas."""
    global _last_emitted_key, _last_emitted_frames

    cfg = {**_default_postprocess_config, **(config or {})}
    groups = _compute_edge_groups(edges, total_emit_frames)

    if not groups:
        return []

    min_support = cfg["min_support_frames"]
    min_consec = cfg["min_consecutive_frames"]
    min_ratio = cfg["min_support_ratio"]
    hysteresis = cfg["hysteresis_margin"]
    allow_multiple = cfg["allow_multiple_events_per_emit"]
    dedup = cfg["deduplicate_events"]

    actions: list[dict[str, Any]] = []
    emitted_this_block: set[tuple[str, str]] = set()

    for g in groups:
        if g.support_frames < min_support:
            continue
        if g.longest_consecutive_run < min_consec:
            continue
        if g.support_ratio < min_ratio:
            continue

        key = (g.canonical_src, g.canonical_dst)

        # hysteresis: don't flip if old edge still has meaningful support
        if dedup and _last_emitted_key is not None and key != _last_emitted_key:
            # check if old key still has support
            old_group = next((gr for gr in groups
                              if (gr.canonical_src, gr.canonical_dst) == _last_emitted_key), None)
            if old_group is not None and old_group.support_frames > g.support_frames * (1 - hysteresis):
                # old edge is still competitive, keep it
                key = _last_emitted_key
                g = old_group

        if dedup and key in emitted_this_block:
            continue
        emitted_this_block.add(key)

        actions.append({
            "frame_id": g.first_frame,
            "start_frame": g.first_frame,
            "end_frame": g.last_frame,
            "canonical_src": g.canonical_src,
            "canonical_dst": g.canonical_dst,
            "edge_src": g.edge_src,
            "edge_dst": g.edge_dst,
            "event_type": g.event_type,
            "player_id": g.canonical_src,
            "receiver_id": g.canonical_dst if not _is_self((g.edge_src, g.edge_dst)) else None,
            "timestamp": str(g.first_frame / 25.0),
            "start_x": None, "start_y": None, "end_x": None, "end_y": None,
            "confidence": None,
            "is_realtime": True,
            "is_final": True,
            "support_frames": g.support_frames,
            "support_ratio": round(g.support_ratio, 3),
            "longest_consecutive_run": g.longest_consecutive_run,
            "emit_block_first_frame": edges[0].get("frame_id", g.first_frame) if edges else g.first_frame,
            "emit_block_last_frame": edges[-1].get("frame_id", g.last_frame) if edges else g.last_frame,
            "source": "postprocess_emit_block",
        })

        if not allow_multiple:
            break

        _last_emitted_key = key
        _last_emitted_frames = g.support_frames

    return actions
