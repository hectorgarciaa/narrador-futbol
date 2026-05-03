from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd

from football_ai.actions.pathcrf_adapter import PathCRFAdapterConfig, PathCRFTracksAdapter
from football_ai.actions.pathcrf_wrapper import (
    PathCRFInferenceConfig,
    PathCRFRenderConfig,
    run_pathcrf_inference,
    run_pathcrf_pipeline,
)
from football_ai.core import PHASE_ACTIONS_DETECTOR, Phase, make_phase_packet
from football_ai.pathcrf_slot_mapping import (
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)

from .postprocess import RealtimeAction, RealtimeCheckpoint, _classify_event
from .incremental_tracking_builder import IncrementalTrackingBuilder, IncrementalBuilderConfig

TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True)
class RollingActionsConfig:
    enabled: bool = True
    fps: float = 25.0
    cadence_frames: int = 10
    min_frames_warmup: int = 50
    emit_delay_frames: int = 100
    emit_frames: int = 10
    repo_path: Path = Path("football_ai/actions/repo/pathcrf")
    trial: int = 120
    model_file: str = "state_dict_best_acc.pt"
    device: str = "auto"
    use_crf: bool = True
    decode: str = "indep"
    window_seconds: float = 10.0
    sample_freq: int = 5
    min_event_duration: int = 10
    smooth_edges: bool = False
    export_debug: bool = True
    output_dir: str | None = None
    async_enabled: bool = False
    max_workers: int = 1
    drop_policy: str = "latest"
    snapshot_window_frames: int | None = None
    incremental_tracking: bool = False
    incremental_tracking_mode: str = "adapter_tail_replace"
    incremental_recompute_tail_frames: int = 75
    incremental_tail_overlap_frames: int = 100
    incremental_validate_every: int = 10
    incremental_validation_tolerance: float = 1e-3
    incremental_fallback_on_mismatch: bool = True


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

        if self.config.async_enabled:
            self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.config.max_workers)))

        self._builder: IncrementalTrackingBuilder | None = None
        self._init_builder()

    @property
    def builder(self) -> IncrementalTrackingBuilder | None:
        return self._builder

    def _init_builder(self) -> None:
        if not self.config.incremental_tracking:
            self._builder = None
            return
        window = self.config.snapshot_window_frames or max(int(self.config.window_seconds * self.config.fps) * 2, 300)
        self._builder = IncrementalTrackingBuilder(
            IncrementalBuilderConfig(
                fps=float(self.config.fps),
                snapshot_window_frames=window,
                recompute_tail_frames=int(self.config.incremental_recompute_tail_frames),
                tail_overlap_frames=int(self.config.incremental_tail_overlap_frames),
                mode=str(self.config.incremental_tracking_mode),
                validate_every=int(self.config.incremental_validate_every),
                validation_tolerance=float(self.config.incremental_validation_tolerance),
                fallback_on_mismatch=bool(self.config.incremental_fallback_on_mismatch),
            )
        )

    def _origin(self) -> int:
        return self._builder.origin if self._builder else 0

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
        self._init_builder()

    # ── Phase execute ──

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

        clean_out = dict(clean_in)
        clean_out["actions_incremental"] = {
            "raw_edge": raw_edge or {},
            "raw_edge_batch": [],
            "confirmed_action": confirmed_action or {},
            "action_metadata": action_metadata,
        }
        trace_out = dict(trace_in)
        trace_out["actions_incremental"] = {
            "mode": "rolling",
            "buffer_frames": self._frame_count(),
            "checkpoints": self._checkpoint_count,
            "emitted_edges": len(self._emitted_edges),
            "last_inference_frame": self._last_inference_frame,
            "last_infer_ms": self._last_infer_ms,
            "async_enabled": self.config.async_enabled,
            "skipped": self._skipped_checkpoints,
            "completed": self._completed_checkpoints,
        }

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
        if cfg.incremental_tracking and self._builder is not None and self._builder.is_ready:
            if cfg.incremental_tracking_mode == "adapter_tail_replace":
                edge_df = self._run_tail_replace_inference(frame_index)
            else:
                edge_df = self._run_incremental_inference(frame_index)
        else:
            edge_df = self._run_full_checkpoint(frame_index)
        timing_ms = (perf_counter() - started) * 1000.0
        return frame_index, edge_df, emit_start, emit_end, timing_ms

    def _run_full_checkpoint(self, frame_index: int) -> pd.DataFrame:
        cfg = self.config
        full_frames = self._frame_count()
        window_size = cfg.snapshot_window_frames or max(int(cfg.window_seconds * cfg.fps) * 2, 300)
        origin = max(0, full_frames - window_size)

        trimmed: dict[str, Any] = {}
        for cls in TRACK_CLASSES:
            data = self._buffer.get(cls, [])
            trimmed[cls] = data[origin:]

        adapter = PathCRFTracksAdapter(PathCRFAdapterConfig(fps=float(cfg.fps)))
        tracking_df, person_slots, referee_slots = adapter.convert_tracks_to_df_with_slots(trimmed)
        self._last_tracking_df = tracking_df
        self._last_person_slots = person_slots
        self._last_referee_slots = referee_slots

        if self._builder is not None:
            self._builder.init_from_full(tracking_df.copy(), origin, self._buffer, full_frames)

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

    def _run_incremental_inference(self, frame_index: int) -> pd.DataFrame:
        cfg = self.config
        full_frames = self._frame_count()

        if self._builder is not None:
            self._builder.extend(self._buffer, full_frames)

        # Periodic validation
        if (self._builder is not None and self._builder.cfg is not None
                and self._checkpoint_count > 1
                and self._checkpoint_count % self._builder.cfg.validate_every == 0):
            full_df, full_origin = self._build_full_tracking_for_validation(frame_index)
            if full_df is not None:
                info = self._builder.validate_and_fallback(full_df, full_origin, self._buffer, full_frames)
                tag = "MISMATCH" if info.get("mismatch") else "OK"
                print(f"[rolling] frame {frame_index} | validate: {tag} ({info.get('differences',0)} cols diff, max {info.get('max_abs_diff',0):.4f})")

        result = run_pathcrf_inference(
            tracking_df=self._builder.tracking_df,
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

    def _build_full_tracking_for_validation(self, frame_index: int):
        cfg = self.config
        full_frames = self._frame_count()
        window_size = cfg.snapshot_window_frames or max(int(cfg.window_seconds * cfg.fps) * 2, 300)
        origin = max(0, full_frames - window_size)

        trimmed = {}
        for cls in TRACK_CLASSES:
            data = self._buffer.get(cls, [])
            trimmed[cls] = data[origin:]

        adapter = PathCRFTracksAdapter(PathCRFAdapterConfig(fps=float(cfg.fps)))
        return adapter.convert_tracks_to_df(trimmed), origin

    def _run_tail_replace_inference(self, frame_index: int) -> pd.DataFrame:
        """Run adapter on tail window, merge with cached head. Returns edge_df."""
        cfg = self.config
        full_frames = self._frame_count()

        snapshot, snapshot_origin, update_start = self._builder.prepare_tail_snapshot(self._buffer, full_frames)

        adapter = PathCRFTracksAdapter(PathCRFAdapterConfig(fps=float(cfg.fps)))
        adapter_df = adapter.convert_tracks_to_df(snapshot)
        self._builder.merge_tail_from_adapter(adapter_df, snapshot_origin, update_start, full_frames)

        # Periodic validation
        builder_cfg = self._builder.cfg if self._builder else None
        if (builder_cfg is not None
                and self._checkpoint_count > 0
                and (self._checkpoint_count + 1) % builder_cfg.validate_every == 0):
            full_df, full_origin = self._build_full_tracking_for_validation(frame_index)
            if full_df is not None:
                info = self._builder.validate_and_fallback(full_df, full_origin, self._buffer, full_frames)
                tag = "MISMATCH" if info.get("mismatch") else "OK"
                print(f"[rolling] frame {frame_index} | validate: {tag} ({info.get('differences',0)} cols diff, max {info.get('max_abs_diff',0):.4f})")

        result = run_pathcrf_inference(
            tracking_df=self._builder.tracking_df,
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

            for _, row in edge_df.iterrows():
                lf = _coerce_int(row.get("frame_id"))
                if lf is None:
                    continue
                global_frame = self._origin() + int(lf)
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

                et = _classify_event(es, ed)
                act = RealtimeAction(
                    frame_id=global_frame, edge_src=es, edge_dst=ed, canonical_src=cs, canonical_dst=cd,
                    event_type=et, player_id=cs, receiver_id=cd if es != ed else None,
                    timestamp=str(global_frame / max(float(self.config.fps), 1e-6)),
                    start_x=None, start_y=None, end_x=None, end_y=None, confidence=None,
                    is_realtime=True, is_final=True,
                )
                self._postprocessed_actions.append(asdict(act))
                emitted_this += 1

                if global_frame > self._latest_emitted_frame:
                    self._latest_emitted_frame = global_frame

                if global_frame == fi:
                    raw_edge_for_frame = {"frame_id": global_frame, "edge_src": es, "edge_dst": ed}
                    confirmed_action_for_frame = {**raw_edge_for_frame, "event_type": et, "player_id": cs,
                                                  "receiver_id": cd if es != ed else None,
                                                  "is_realtime": True, "is_final": True}

            self._checkpoints.append(RealtimeCheckpoint(
                checkpoint_frame=int(fi), snapshot_frames=self._frame_count(),
                edge_rows=int(len(edge_df)), emit_start=int(emit_start), emit_end=int(emit_end),
                emitted_count=emitted_this, infer_ms=round(timing_ms, 1),
                total_emitted_so_far=len(self._emitted_edges),
            ))

            tracking_len = len(self._builder.tracking_df) if self._builder and self._builder.tracking_df is not None else "full"
            print(f"[rolling] frame {fi} | tracking={tracking_len} edges={len(edge_df)} emit=[{emit_start},{emit_end}] emitted={emitted_this} infer_ms={timing_ms:.0f}")

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
            "emitted_edges": output_dir / "emitted_edges.parquet",    # keep path for backward compat
            "rolling_edges": output_dir / "rolling_edges_per_frame.parquet",
            "postprocessed_actions": output_dir / "postprocessed_actions.parquet",
            "checkpoints": output_dir / "runtime_checkpoints.json",
            "summary": output_dir / "summary.json",
        }

        if self.config.export_debug:
            emitted_df.to_parquet(result["emitted_edges"], index=False)
            rolling_df.to_parquet(result["rolling_edges"], index=False)
            postproc_df.to_parquet(result["postprocessed_actions"], index=False)

            builder_stats = self._builder.stats() if self._builder else {}
            chk_recs = [asdict(cp) for cp in self._checkpoints]
            times = [c.infer_ms for c in self._checkpoints]
            import numpy as np
            with result["checkpoints"].open("w") as f:
                json.dump({
                    "mode": "rolling", "incremental": self.config.incremental_tracking,
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
                    "builder_stats": builder_stats,
                }, f, ensure_ascii=False, indent=2)

            with result["summary"].open("w") as f:
                json.dump({
                    "mode": "rolling", "incremental": self.config.incremental_tracking,
                    "config": {"fps": float(self.config.fps), "cadence_frames": self.config.cadence_frames,
                               "emit_delay_frames": self.config.emit_delay_frames, "emit_frames": self.config.emit_frames,
                               "async_enabled": self.config.async_enabled},
                    "frame_count": self._frame_count(), "num_checkpoints": len(chk_recs),
                    "total_emitted_edges": len(self._emitted_edges),
                    "total_rolling_edges": len(self._rolling_edges),
                    "avg_infer_ms": round(self._total_infer_ms / max(1, self._completed_checkpoints), 1),
                    "builder_stats": builder_stats,
                }, f, ensure_ascii=False, indent=2)

        return result

    def _checkpoint_output_dir(self) -> Path:
        if self.config.output_dir:
            return Path(self.config.output_dir) / "snapshots"
        return Path("output/actions/rolling_online/snapshots")
