from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd

from .adapter import PathCRFAdapterConfig, PathCRFTracksAdapter
from .inference import PathCRFInferenceConfig, run_pathcrf_inference
from .postprocess import (
    RealtimeCheckpoint,
    _classify_event,
    postprocess_emit_block,
    reset_postprocess_state,
)
from football_ai.core import PHASE_ACTIONS_DETECTOR, Phase, make_phase_packet
from football_ai.pathcrf_slot_mapping import (
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)

TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True)
class RollingActionsConfig:
    enabled: bool = True
    fps: float = 25.0
    cadence_frames: int = 40
    min_frames_warmup: int = 50
    emit_delay_frames: int = 50
    emit_frames: int = 40
    repo_path: Path = Path("external/pathcrf")
    trial: int = 120
    model_file: str = "state_dict_best_acc.pt"
    device: str = "auto"
    use_crf: bool = True
    decode: str = "indep"
    window_seconds: float = 10.0
    sample_freq: int = 5
    min_event_duration: int = 10
    smooth_edges: bool = False
    export_debug: bool = False
    output_dir: str | None = None
    async_enabled: bool = True
    max_workers: int = 1
    drop_policy: str = "latest"
    snapshot_window_frames: int | None = 750


def _slot_to_canonical(slot_name: Any) -> str | None:
    if slot_name is None:
        return None
    text = str(slot_name).strip()
    if not text:
        return None
    person = person_slot_to_canonical_id(text)
    if person is not None:
        return str(person)
    referee = referee_slot_to_canonical_id(text)
    if referee is not None:
        return str(referee)
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, pd.Timestamp):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class RollingActionsPhase(Phase):
    def __init__(self, config: RollingActionsConfig | None = None):
        self.config = config or RollingActionsConfig()
        self._buffer: dict[str, Any] = {}
        self._emitted_edges: list[dict[str, Any]] = []
        self._rolling_edges: list[dict[str, Any]] = []
        self._postprocessed_actions: list[dict[str, Any]] = []
        self._checkpoints: list[RealtimeCheckpoint] = []
        self._checkpoint_count = 0
        self._total_infer_ms = 0.0
        self._last_inference_frame = -10**9
        self._latest_edge_by_frame: dict[int, dict[str, Any]] = {}
        self._skipped_checkpoints = 0
        self._completed_checkpoints = 0
        self._last_infer_ms: float = 0.0
        self._latest_emitted_frame = -1
        self._last_tracking_df: pd.DataFrame | None = None
        self._last_person_slots: dict[str, str] = {}
        self._last_referee_slots: dict[str, str] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._pending_job: Future | None = None
        self._pending_frame_index: int | None = None
        self._lock = threading.Lock()
        self._finalized = False
        self._snapshot_cache: dict[str, Any] | None = None
        self._snapshot_frame_index: int | None = None
        self._last_checkpoint_result: dict[str, Any] | None = None

        if self.config.async_enabled:
            self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.config.max_workers)))

    @property
    def builder(self) -> None:
        return None

    def reset(self) -> None:
        self._buffer = {}
        self._emitted_edges = []
        self._rolling_edges = []
        self._postprocessed_actions = []
        self._checkpoints = []
        self._checkpoint_count = 0
        self._total_infer_ms = 0.0
        self._last_inference_frame = -10**9
        self._latest_edge_by_frame = {}
        self._skipped_checkpoints = 0
        self._completed_checkpoints = 0
        self._last_infer_ms = 0.0
        self._latest_emitted_frame = -1
        self._last_tracking_df = None
        self._last_person_slots = {}
        self._last_referee_slots = {}
        self._pending_job = None
        self._pending_frame_index = None
        self._finalized = False
        self._snapshot_cache = None
        self._snapshot_frame_index = None
        self._last_checkpoint_result = None
        reset_postprocess_state()

    def execute(self, position_packet: dict) -> dict:
        clean_in = dict(position_packet["clean"])
        trace_in = dict(position_packet["trace"])
        frame_index = int(position_packet["frame_index"])

        tracks_frame = clean_in.get("tracks_frame", {})
        self._accumulate_frame(tracks_frame, frame_index)

        should_infer = (
            (frame_index + 1) >= int(self.config.min_frames_warmup)
            and (
                self._last_inference_frame < 0
                or (frame_index - self._last_inference_frame) >= int(self.config.cadence_frames)
            )
        )

        raw_edge = None
        confirmed_action = None
        action_metadata: dict[str, Any] = {"mode": "rolling", "should_infer": should_infer}

        if should_infer:
            self._last_inference_frame = frame_index
            if self._executor is not None:
                raw_edge, confirmed_action, action_metadata = self._run_checkpoint_async(frame_index)
            else:
                raw_edge, confirmed_action, action_metadata = self._run_checkpoint_sync(frame_index)

        if raw_edge is None and self._latest_edge_by_frame:
            latest = self._latest_edge_by_frame.get(frame_index)
            if latest:
                raw_edge = {"frame_id": frame_index, "edge_src": latest["edge_src"], "edge_dst": latest["edge_dst"]}
                event_type = _classify_event(raw_edge.get("edge_src"), raw_edge.get("edge_dst"))
                confirmed_action = {**raw_edge, "event_type": event_type, "is_realtime": True, "is_final": True}
                action_metadata = {"mode": "rolling", "cached": True, "frame_id": frame_index}

        if confirmed_action is None:
            confirmed_action = {}

        # ── Build unified actions_packet ──
        chk = self._last_checkpoint_result or {}
        chk_completed = bool(chk.get("checkpoint_completed"))

        actions_packet = {
            "frame_id": frame_index,
            "checkpoint_frame": chk.get("checkpoint_frame"),
            "checkpoint_completed": chk_completed,
            "mode": "rolling",
            "snapshot_start": chk.get("snapshot_start"),
            "snapshot_end": chk.get("snapshot_end"),
            "emit_start": chk.get("emit_start"),
            "emit_end": chk.get("emit_end"),
            "emitted_edges": chk.get("emitted_edges") or [],
            "emitted_events": chk.get("emitted_events") or [],
            "latest_edge": chk.get("latest_edge"),
            "latest_event": chk.get("latest_event"),
            "async_pending": chk.get("async_pending", False),
            "skipped": chk.get("skipped", False),
            "dropped": chk.get("dropped", False),
            "timings": chk.get("timings") or {},
            "metadata": action_metadata,
            # Per-frame data (always present)
            "raw_edge": raw_edge or {},
            "confirmed_action": confirmed_action or {},
            "slot_mappings": {
                "person_slots": {str(k): str(v) for k, v in (self._last_person_slots or {}).items()},
                "referee_slots": {str(k): str(v) for k, v in (self._last_referee_slots or {}).items()},
            },
            "buffer_frames": self._frame_count(),
            "total_checkpoints": self._checkpoint_count,
            "total_emitted_edges": len(self._emitted_edges),
            "total_completed": self._completed_checkpoints,
            "total_skipped": self._skipped_checkpoints,
            "last_infer_ms": self._last_infer_ms,
            "async_enabled": self.config.async_enabled,
        }
        self._last_checkpoint_result = None

        clean_out = dict(clean_in)
        clean_out["actions_packet"] = actions_packet
        trace_out = dict(trace_in)

        return make_phase_packet(
            phase_name=PHASE_ACTIONS_DETECTOR,
            frame_index=frame_index,
            frame_time_ms=position_packet["frame_time_ms"],
            image_width=position_packet["image_width"],
            image_height=position_packet["image_height"],
            clean=clean_out,
            trace=trace_out,
        )

    def _frame_count(self) -> int:
        return max((len(self._buffer.get(k, [])) for k in TRACK_CLASSES), default=0)

    def _accumulate_frame(self, tracks_frame: Mapping[str, Any], frame_index: int) -> None:
        if not self._buffer:
            for cls in TRACK_CLASSES:
                self._buffer[cls] = []
        for cls in TRACK_CLASSES:
            frame_data = tracks_frame.get(cls, {})
            self._buffer[cls].append(frame_data if isinstance(frame_data, dict) else {})

    def _make_snapshot_copy(self) -> dict[str, Any]:
        return {cls: list(self._buffer.get(cls, [])) for cls in TRACK_CLASSES}

    # ── Checkpoint dispatch ──

    def _run_checkpoint_sync(self, frame_index: int):
        result = self._execute_checkpoint(frame_index)
        return self._on_checkpoint_done(result, frame_index)

    def _run_checkpoint_async(self, frame_index: int):
        self._collect_pending_results()

        if self._pending_job is not None and not self._pending_job.done():
            if self.config.drop_policy == "skip":
                self._skipped_checkpoints += 1
                print(f"[rolling] frame {frame_index} | SKIPPED")
                return self._latest_edge_or_none(frame_index)
            self._snapshot_cache = self._make_snapshot_copy()
            self._snapshot_frame_index = frame_index
            self._skipped_checkpoints += 1
            print(f"[rolling] frame {frame_index} | QUEUED (latest policy)")
            return self._latest_edge_or_none(frame_index)

        self._snapshot_cache = None
        self._snapshot_frame_index = None
        self._pending_job = self._executor.submit(self._execute_checkpoint, frame_index)
        self._pending_frame_index = frame_index
        print(f"[rolling] frame {frame_index} | async job submitted")
        return self._latest_edge_or_none(frame_index)

    def _collect_pending_results(self) -> None:
        if self._pending_job is None or not self._pending_job.done():
            return
        try:
            result = self._pending_job.result()
            if result is not None:
                self._on_checkpoint_done(result, self._pending_frame_index or 0)
        except Exception as e:
            print(f"[rolling] checkpoint job at frame {self._pending_frame_index} failed: {e}")
        finally:
            self._pending_job = None
        if self._snapshot_cache is not None:
            deferred, deferred_frame = self._snapshot_cache, self._snapshot_frame_index
            self._snapshot_cache = None
            self._snapshot_frame_index = None
            if self._executor is not None:
                self._pending_job = self._executor.submit(self._execute_checkpoint, deferred_frame)
                self._pending_frame_index = deferred_frame
                print(f"[rolling] deferred job at frame {deferred_frame} submitted")

    def _latest_edge_or_none(self, frame_index: int):
        latest = self._latest_edge_by_frame.get(frame_index)
        if latest:
            raw_edge = {"frame_id": frame_index, "edge_src": latest["edge_src"], "edge_dst": latest["edge_dst"]}
            et = _classify_event(raw_edge.get("edge_src"), raw_edge.get("edge_dst"))
            return raw_edge, {**raw_edge, "event_type": et, "is_realtime": True, "is_final": True}, {"mode": "rolling", "cached": True}
        return None, None, {"mode": "rolling", "no_edge_yet": True}

    # ── Execution ──

    def _execute_checkpoint(self, frame_index: int):
        cfg = self.config
        emit_end = frame_index - int(cfg.emit_delay_frames)
        emit_start = emit_end - int(cfg.emit_frames) + 1
        if emit_start > emit_end:
            emit_start = emit_end

        started = perf_counter()
        edge_df = self._run_full_checkpoint(frame_index)
        timing_ms = (perf_counter() - started) * 1000.0
        return frame_index, edge_df, emit_start, emit_end, timing_ms

    def _run_full_checkpoint(self, frame_index: int) -> pd.DataFrame:
        cfg = self.config
        full_frames = self._frame_count()
        window_size = cfg.snapshot_window_frames or max(int(cfg.window_seconds * cfg.fps) * 2, 300)
        origin = max(0, full_frames - window_size)

        self._last_snapshot_start = origin
        self._last_snapshot_end = full_frames - 1

        trimmed: dict[str, Any] = {}
        for cls in TRACK_CLASSES:
            data = self._buffer.get(cls, [])
            trimmed[cls] = data[origin:]

        adapter = PathCRFTracksAdapter(PathCRFAdapterConfig(fps=float(cfg.fps)))
        tracking_df, person_slots, referee_slots = adapter.convert_tracks_to_df_with_slots(trimmed)
        self._last_tracking_df = tracking_df
        self._last_person_slots = person_slots
        self._last_referee_slots = referee_slots

        result = run_pathcrf_inference(
            tracking_df=tracking_df,
            return_df=True,
            config=PathCRFInferenceConfig(
                repo_path=cfg.repo_path, trial=int(cfg.trial), model_file=str(cfg.model_file),
                use_crf=bool(cfg.use_crf), decode=str(cfg.decode),
                window_seconds=float(cfg.window_seconds), fps=float(cfg.fps),
                sample_freq=int(cfg.sample_freq), min_event_duration=int(cfg.min_event_duration),
                device=str(cfg.device),
            ),
        )
        return result.edge_sequence_df

    # ── Result processing ──

    def _on_checkpoint_done(self, result_tuple, frame_index: int = 0):
        with self._lock:
            if isinstance(result_tuple, tuple) and len(result_tuple) == 5:
                fi, edge_df, emit_start, emit_end, timing_ms = result_tuple
            else:
                return self._latest_edge_or_none(frame_index)

            self._total_infer_ms += timing_ms
            self._last_infer_ms = timing_ms
            self._checkpoint_count += 1
            self._completed_checkpoints += 1

            if "frame_id" not in edge_df.columns:
                edge_df = edge_df.reset_index()
                if "frame_id" not in edge_df.columns and "index" in edge_df.columns:
                    edge_df = edge_df.rename(columns={"index": "frame_id"})
                elif "frame_id" not in edge_df.columns:
                    edge_df["frame_id"] = range(len(edge_df))

            emitted_this = 0
            raw_edge_for_frame = None
            confirmed_action_for_frame = None
            emit_block_edges: list[dict[str, Any]] = []

            for _, row in edge_df.iterrows():
                lf = _coerce_int(row.get("frame_id"))
                if lf is None:
                    continue
                global_frame = int(lf)
                es = str(row.get("edge_src")) if row.get("edge_src") is not None and not pd.isna(row.get("edge_src")) else None
                ed = str(row.get("edge_dst")) if row.get("edge_dst") is not None and not pd.isna(row.get("edge_dst")) else None
                cs, cd = _slot_to_canonical(es), _slot_to_canonical(ed)

                rec = {"frame_id": global_frame, "checkpoint_frame": int(fi), "edge_src": es, "edge_dst": ed,
                       "canonical_src": cs, "canonical_dst": cd}
                self._rolling_edges.append(rec)
                self._latest_edge_by_frame[global_frame] = rec

                if not (emit_start <= global_frame <= emit_end):
                    continue

                emit_rec = dict(rec)
                emit_rec["emit_start_frame"] = int(emit_start)
                emit_rec["emit_end_frame"] = int(emit_end)
                emit_rec["latency_frames"] = int(fi) - global_frame
                emit_rec["latency_seconds"] = float(emit_rec["latency_frames"]) / max(float(self.config.fps), 1e-6)
                self._emitted_edges.append(emit_rec)
                emit_block_edges.append(emit_rec)
                emitted_this += 1

                if global_frame > self._latest_emitted_frame:
                    self._latest_emitted_frame = global_frame

                if global_frame == fi:
                    raw_edge_for_frame = {"frame_id": global_frame, "edge_src": es, "edge_dst": ed}
                    confirmed_action_for_frame = {**raw_edge_for_frame, "event_type": _classify_event(es, ed),
                                                  "player_id": cs, "receiver_id": cd if es != ed else None,
                                                  "is_realtime": True, "is_final": True}

            # Consolidar acciones del bloque emitido con postprocesado temporal
            emit_frame_count = int(self.config.emit_frames)
            consolidated = postprocess_emit_block(emit_block_edges, emit_frame_count)
            self._postprocessed_actions.extend(consolidated)

            self._checkpoints.append(RealtimeCheckpoint(
                checkpoint_frame=int(fi), snapshot_frames=self._frame_count(),
                edge_rows=int(len(edge_df)), emit_start=int(emit_start), emit_end=int(emit_end),
                emitted_count=emitted_this, infer_ms=round(timing_ms, 1),
                total_emitted_so_far=len(self._emitted_edges),
            ))

            print(f"[rolling] frame {fi} | tracking=full edges={len(edge_df)} emit=[{emit_start},{emit_end}] emitted={emitted_this} infer_ms={timing_ms:.0f}")

            # Populate checkpoint result for the enriched packet
            latest_event = consolidated[0] if consolidated else None
            latest_edge_emitted = emit_block_edges[-1] if emit_block_edges else None
            self._last_checkpoint_result = {
                "checkpoint_completed": True,
                "checkpoint_frame": int(fi),
                "snapshot_start": getattr(self, "_last_snapshot_start", 0),
                "snapshot_end": getattr(self, "_last_snapshot_end", self._frame_count() - 1),
                "emit_start": int(emit_start),
                "emit_end": int(emit_end),
                "emitted_edges": emit_block_edges,
                "emitted_events": consolidated,
                "latest_edge": latest_edge_emitted,
                "latest_event": latest_event,
                "async_pending": False,
                "skipped": False,
                "dropped": False,
                "timings": {
                    "total_checkpoint_ms": round(timing_ms, 1),
                },
            }

            return raw_edge_for_frame, confirmed_action_for_frame, {
                "mode": "rolling", "should_infer": True, "frame_id": fi, "checkpoint_frame": fi,
                "snapshot_frames": self._frame_count(), "edge_rows": len(edge_df),
                "emit_start": int(emit_start), "emit_end": int(emit_end),
                "emitted_this_checkpoint": emitted_this, "infer_ms": round(timing_ms, 1),
                "total_emitted": len(self._emitted_edges), "total_infer_ms": round(self._total_infer_ms, 1),
            }

    # ── Finalize ──

    def finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        if self._executor is not None:
            if self._pending_job is not None:
                if not self._pending_job.done():
                    print("[rolling] waiting for pending checkpoint job to finish...")
                try:
                    result = self._pending_job.result(timeout=120)
                    if result is not None:
                        self._on_checkpoint_done(result, self._pending_frame_index or 0)
                except Exception as e:
                    print(f"[rolling] pending job failed: {e}")
                finally:
                    self._pending_job = None
            if self._snapshot_cache is not None:
                deferred, deferred_frame = self._snapshot_cache, self._snapshot_frame_index
                self._snapshot_cache = None
                self._snapshot_frame_index = None
                self._on_checkpoint_done(self._execute_checkpoint(deferred_frame or 0), deferred_frame or 0)
            self._executor.shutdown(wait=True)
            self._executor = None

    def build_result(self, output_dir: str | Path) -> dict[str, Any]:
        self.finalize()
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        emitted_df = pd.DataFrame(self._emitted_edges)
        postproc_df = pd.DataFrame(self._postprocessed_actions)
        rolling_df = pd.DataFrame(self._rolling_edges)

        result: dict[str, Any] = {
            "emitted_edges_df": emitted_df,
            "postprocessed_actions_df": postproc_df,
            "emitted_edges": output_dir / "emitted_edges.parquet",
            "rolling_edges": output_dir / "rolling_edges_per_frame.parquet",
            "postprocessed_actions": output_dir / "postprocessed_actions.parquet",
            "checkpoints": output_dir / "runtime_checkpoints.json",
            "summary": output_dir / "summary.json",
        }

        if self.config.export_debug:
            emitted_df.to_parquet(result["emitted_edges"], index=False)
            rolling_df.to_parquet(result["rolling_edges"], index=False)
            postproc_df.to_parquet(result["postprocessed_actions"], index=False)

            chk_recs = [asdict(cp) for cp in self._checkpoints]
            times = [c.infer_ms for c in self._checkpoints]
            import numpy as np
            with result["checkpoints"].open("w") as f:
                json.dump({
                    "mode": "rolling",
                    "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.config).items()},
                    "checkpoints": chk_recs, "checkpoint_count": len(chk_recs),
                    "completed": self._completed_checkpoints, "skipped": self._skipped_checkpoints,
                    "emitted_edges": len(self._emitted_edges), "rolling_edges": len(self._rolling_edges),
                    "postprocessed": len(self._postprocessed_actions),
                    "total_infer_ms": round(self._total_infer_ms, 1),
                    "avg_infer_ms": round(self._total_infer_ms / max(1, self._completed_checkpoints), 1),
                    "p95_infer_ms": round(float(np.percentile(times, 95)), 1) if times else 0,
                    "max_infer_ms": round(float(np.max(times)), 1) if times else 0,
                    "latest_emitted_frame": self._latest_emitted_frame,
                }, f, ensure_ascii=False, indent=2)

            with result["summary"].open("w") as f:
                json.dump({
                    "mode": "rolling",
                    "config": {"fps": float(self.config.fps), "cadence_frames": self.config.cadence_frames,
                               "emit_delay_frames": self.config.emit_delay_frames, "emit_frames": self.config.emit_frames,
                               "async_enabled": self.config.async_enabled},
                    "frame_count": self._frame_count(), "num_checkpoints": len(chk_recs),
                    "total_emitted_edges": len(self._emitted_edges),
                    "total_rolling_edges": len(self._rolling_edges),
                    "avg_infer_ms": round(self._total_infer_ms / max(1, self._completed_checkpoints), 1),
                }, f, ensure_ascii=False, indent=2)

        return result
