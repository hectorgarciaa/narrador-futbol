from __future__ import annotations

import json
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from football_ai.core import Phase, get_logger, make_phase_packet
from football_ai.core.phase_packets import PHASE_COMMENTARY
from football_ai.commentaries.generator import CommentaryEvent, CommentaryGenerator
from football_ai.commentaries.voice import CommentaryAudioPipeline, build_voice_synthesizer
from football_ai.actions.pathcrf_commentary import (
    COMMENTARY_ACTION_MAP,
    FIELD_ZONE_LABELS,
    TEAM_IN_FAVOR_ACTIONS,
)
from football_ai.actions.pathcrf_semantics import (
    PITCH_LENGTH_M,
    team_id_from_player_id,
    team_attacks_right,
)

logger = get_logger(__name__)


@dataclass
class CommentaryPhaseConfig:
    enabled: bool = False

    generate_text: bool = True
    generate_audio: bool = True

    llm_model: str = "gemma4-q4ks-text"
    llm_base_url: str | None = None
    llm_temperature: float = 0.7

    tts_backend: str = "xtts"
    speaker_wavs: list[str] | None = None
    female_speaker_wavs: list[str] | None = None
    alternate_voices: bool = False

    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_female_voice_id: str | None = None
    elevenlabs_model_id: str | None = None
    elevenlabs_output_format: str | None = None
    elevenlabs_language_code: str | None = None
    elevenlabs_stability: float | None = None
    elevenlabs_similarity_boost: float | None = None
    elevenlabs_style: float | None = None
    elevenlabs_speed: float | None = None
    elevenlabs_use_speaker_boost: bool | None = None
    elevenlabs_optimize_streaming_latency: int | None = None

    fps: float = 25.0
    output_dir: str | None = None
    manifest_filename: str = "commentary_manifest.jsonl"

    max_workers: int = 2
    drop_policy: str = "latest"
    max_queue_size: int = 8

    skip_event_types: list[str] = field(default_factory=lambda: ["out", "unknown"])
    deduplicate_consecutive: bool = True


def _coerce_int_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip() or None


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _field_zone_key(field_position_m: tuple[float, float] | None, *, attacks_right: bool) -> str | None:
    if field_position_m is None:
        return None
    x_m = max(0.0, min(PITCH_LENGTH_M, float(field_position_m[0])))
    first_third = PITCH_LENGTH_M / 3.0
    second_third = (2.0 * PITCH_LENGTH_M) / 3.0
    if attacks_right:
        if x_m < first_third:
            return "iniciacion"
        if x_m < second_third:
            return "creacion"
        return "finalizacion"
    if x_m < first_third:
        return "finalizacion"
    if x_m < second_third:
        return "creacion"
    return "iniciacion"


class CommentaryPhase(Phase):
    def __init__(self, config: CommentaryPhaseConfig | None = None) -> None:
        self.config = config or CommentaryPhaseConfig()
        self._generator: CommentaryGenerator | None = None
        self._audio_pipeline: CommentaryAudioPipeline | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._pending_job: Future | None = None
        self._queue: deque[dict[str, Any]] = deque()
        self._completed: list[dict[str, Any]] = []
        self._failed: int = 0
        self._track_identity_cache: dict[str, dict[str, Any]] = {}
        self._action_index: int = 0
        self._last_event_signature: str | None = None
        self._output_dir: Path | None = None
        self._manifest_path: Path | None = None
        self._lock = threading.Lock()
        self._finalized: bool = False

        if self.config.enabled:
            self._output_dir = (
                Path(self.config.output_dir).expanduser().resolve()
                if self.config.output_dir
                else Path("output/commentary")
            )
            self._output_dir.mkdir(parents=True, exist_ok=True)
            manifest_filename = str(self.config.manifest_filename or "commentary_manifest.jsonl").strip()
            self._manifest_path = self._output_dir / manifest_filename
            if self.config.generate_audio or self.config.generate_text:
                self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.config.max_workers)))

    def reset(self) -> None:
        self._completed = []
        self._failed = 0
        self._queue.clear()
        self._track_identity_cache = {}
        self._action_index = 0
        self._last_event_signature = None
        self._finalized = False
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            if self.config.enabled and (self.config.generate_audio or self.config.generate_text):
                self._executor = ThreadPoolExecutor(max_workers=max(1, int(self.config.max_workers)))
        self._pending_job = None

    def execute(self, phase_packet: dict) -> dict:
        clean_in = dict(phase_packet["clean"])
        trace_in = dict(phase_packet["trace"])
        frame_index = int(phase_packet["frame_index"])

        if not self.config.enabled:
            clean_out = dict(clean_in)
            clean_out["commentary_packet"] = {"enabled": False}
            return make_phase_packet(
                phase_name=PHASE_COMMENTARY,
                frame_index=frame_index,
                frame_time_ms=phase_packet["frame_time_ms"],
                image_width=phase_packet["image_width"],
                image_height=phase_packet["image_height"],
                clean=clean_out,
                trace=trace_in,
            )

        actions_pkt = clean_in.get("actions_packet", {})

        self._update_identity_cache(clean_in)

        if actions_pkt.get("checkpoint_completed"):
            emitted_events = actions_pkt.get("emitted_events") or []
            slot_mappings = actions_pkt.get("slot_mappings", {})

            for event in emitted_events:
                if not isinstance(event, dict):
                    continue
                item = self._build_commentary_item(event, frame_index, clean_in, slot_mappings)
                if item is None:
                    continue

                if self.config.deduplicate_consecutive:
                    sig = self._event_signature(item["event"])
                    if sig == self._last_event_signature:
                        continue
                    self._last_event_signature = sig

                self._enqueue(item)

        self._collect_completed_jobs()

        with self._lock:
            queued = len(self._queue)
            pending = int(self._pending_job is not None and not self._pending_job.done())
            running = pending
            completed_count = len(self._completed)
            failed_count = self._failed
            last_text = self._completed[-1].get("text", None) if self._completed else None
            last_audio = self._completed[-1].get("audio_path", None) if self._completed else None

        commentary_packet = {
            "enabled": True,
            "queued": queued,
            "running": running,
            "completed": completed_count,
            "failed": failed_count,
            "last_commentary_text": last_text,
            "last_audio_path": last_audio,
        }

        clean_out = dict(clean_in)
        clean_out["commentary_packet"] = commentary_packet
        trace_out = dict(trace_in)

        return make_phase_packet(
            phase_name=PHASE_COMMENTARY,
            frame_index=frame_index,
            frame_time_ms=phase_packet["frame_time_ms"],
            image_width=phase_packet["image_width"],
            image_height=phase_packet["image_height"],
            clean=clean_out,
            trace=trace_out,
        )

    def _update_identity_cache(self, clean_packet: dict) -> None:
        tracks_frame = clean_packet.get("tracks_frame", {})
        for class_name in ("player", "goalkeeper"):
            for raw_track_id, payload in tracks_frame.get(class_name, {}).items():
                if not isinstance(payload, dict):
                    continue
                track_id = str(raw_track_id)
                self._track_identity_cache[track_id] = {
                    "track_id": track_id,
                    "team_name": _clean_str(payload.get("team")),
                    "player_name": _clean_str(payload.get("player_name")),
                    "player_position": self._normalize_position(payload),
                    "lineup_slot": _clean_str(payload.get("lineup_slot")),
                }

    @staticmethod
    def _normalize_position(payload: dict) -> str | None:
        for key in ("lineup_slot", "predicted_role", "predicted_role_frame"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        return None

    def _build_commentary_item(
        self,
        event: dict,
        frame_index: int,
        clean_packet: dict,
        slot_mappings: dict,
    ) -> dict | None:
        raw_type = str(event.get("event_type") or "").strip()
        commentary_action = COMMENTARY_ACTION_MAP.get(raw_type)
        if not commentary_action or commentary_action in self.config.skip_event_types:
            return None

        canonical_src = _coerce_int_str(event.get("canonical_src") or event.get("player_id"))
        canonical_dst = _coerce_int_str(event.get("canonical_dst") or event.get("receiver_id"))
        player_track_id = _coerce_int_str(event.get("player_track_id"))
        receiver_track_id = _coerce_int_str(event.get("receiver_track_id"))

        if player_track_id is None and canonical_src is not None:
            player_track_id = self._resolve_track_id_from_canonical(
                canonical_src, slot_mappings, clean_packet
            )
        if receiver_track_id is None and canonical_dst is not None and canonical_src != canonical_dst:
            receiver_track_id = self._resolve_track_id_from_canonical(
                canonical_dst, slot_mappings, clean_packet
            )

        identity = self._track_identity_cache.get(str(player_track_id)) if player_track_id else None
        receiver_identity = self._track_identity_cache.get(str(receiver_track_id)) if receiver_track_id else None

        team_name = (identity or {}).get("team_name")
        receiver_team_name = (receiver_identity or {}).get("team_name")

        all_team_names = sorted(
            {
                str(id_payload.get("team_name")).strip()
                for id_payload in self._track_identity_cache.values()
                if str(id_payload.get("team_name") or "").strip()
            }
        )
        opponent_team_name = None
        for candidate in all_team_names:
            if candidate != team_name:
                opponent_team_name = candidate
                break

        player_name = (identity or {}).get("player_name") or f"Jugador {player_track_id or canonical_src or '?'}"
        receiver_name = (receiver_identity or {}).get("player_name")
        player_position = (identity or {}).get("player_position") or "JUG"

        field_position_m = self._field_position_from_event(event, clean_packet, player_track_id)
        pathcrf_team_id = team_id_from_player_id(canonical_src)
        attacks_right = team_attacks_right(event.get("period_id"), pathcrf_team_id, None)
        zone_key = _field_zone_key(field_position_m, attacks_right=bool(attacks_right))
        field_zone = FIELD_ZONE_LABELS.get(zone_key) if zone_key else None

        effective_frame_id = int(event.get("start_frame", event.get("frame_id", frame_index)))
        event_time_s = float(effective_frame_id) / max(float(self.config.fps), 1e-6)

        self._action_index += 1

        commentary_event_dict = {
            "action": commentary_action,
            "event_time_s": event_time_s,
            "player_name": str(player_name),
            "player_position": str(player_position),
            "team_name": team_name,
            "opponent_team_name": opponent_team_name,
            "team_in_favor": team_name if commentary_action in TEAM_IN_FAVOR_ACTIONS else None,
            "field_zone": field_zone,
            "action_target": receiver_name,
            "action_index": int(self._action_index),
        }

        metadata = {
            "action_index": int(self._action_index),
            "frame_id": int(effective_frame_id),
            "canonical_src": canonical_src,
            "canonical_dst": canonical_dst,
            "player_track_id": player_track_id,
            "receiver_track_id": receiver_track_id,
            "event_type_raw": raw_type,
            "field_position_m": (
                [float(field_position_m[0]), float(field_position_m[1])]
                if field_position_m is not None
                else None
            ),
            "field_zone_key": zone_key,
            "field_zone": field_zone,
            "attack_direction_right": bool(attacks_right),
            "pathcrf_team_id": pathcrf_team_id,
        }

        return {"event": commentary_event_dict, "metadata": metadata}

    def _resolve_track_id_from_canonical(
        self, canonical_id: str, slot_mappings: dict, clean_packet: dict
    ) -> str | None:
        person_slots = slot_mappings.get("person_slots", {})
        referee_slots = slot_mappings.get("referee_slots", {})
        all_slots = {**person_slots, **referee_slots}
        slot_to_canonical = {str(v): str(k) for k, v in all_slots.items()}
        slot_name = slot_to_canonical.get(str(canonical_id))
        if slot_name is None:
            return None
        tracks_frame = clean_packet.get("tracks_frame", {})
        for class_name in ("player", "goalkeeper", "referee"):
            for track_id, payload in tracks_frame.get(class_name, {}).items():
                if not isinstance(payload, dict):
                    continue
                player_slot = str(payload.get("lineup_slot") or "").strip()
                if player_slot == slot_name:
                    return str(track_id)
        return None

    def _field_position_from_event(
        self, event: dict, clean_packet: dict, track_id: str | None
    ) -> tuple[float, float] | None:
        try:
            return (float(event["start_x"]), float(event["start_y"]))
        except (TypeError, ValueError, KeyError):
            pass
        if track_id is None:
            return None
        tracks_frame = clean_packet.get("tracks_frame", {})
        for class_name in ("player", "goalkeeper"):
            payload = tracks_frame.get(class_name, {}).get(track_id)
            if not isinstance(payload, dict):
                continue
            raw_position = payload.get("field_position_m") or payload.get("field_position")
            if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2:
                try:
                    return (float(raw_position[0]), float(raw_position[1]))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _event_signature(event_dict: dict) -> str:
        return (
            f"{event_dict.get('action')}|"
            f"{event_dict.get('player_name')}|"
            f"{event_dict.get('action_target')}|"
            f"{event_dict.get('event_time_s')}"
        )

    def _enqueue(self, item: dict) -> None:
        with self._lock:
            dp = self.config.drop_policy
            if dp == "skip":
                if self._pending_job is not None and not self._pending_job.done():
                    return
            elif dp == "queue":
                if len(self._queue) >= self.config.max_queue_size:
                    return
                self._queue.append(item)
            elif dp == "latest":
                if len(self._queue) >= self.config.max_queue_size:
                    self._queue.popleft()
                self._queue.append(item)

            if dp in ("queue", "latest"):
                self._maybe_dispatch_next()

    def _maybe_dispatch_next(self) -> None:
        if self._executor is None:
            return
        if self._pending_job is not None and not self._pending_job.done():
            return
        if not self._queue:
            return
        item = self._queue.popleft()
        self._pending_job = self._executor.submit(self._generate_job, item)

    def _generate_job(self, item: dict) -> dict[str, Any]:
        event_dict = item["event"]
        metadata = item["metadata"]

        result: dict[str, Any] = {
            "action_index": metadata.get("action_index"),
            "frame_id": metadata.get("frame_id"),
            "event": event_dict,
            "metadata": metadata,
            "text": None,
            "audio_path": None,
            "audio_duration_seconds": None,
            "llm_seconds": None,
            "tts_seconds": None,
            "error": None,
        }

        try:
            commentary_event = CommentaryEvent.from_dict(event_dict)

            if self._generator is None:
                self._generator = CommentaryGenerator(
                    model=self.config.llm_model,
                    base_url=self.config.llm_base_url,
                    temperature=self.config.llm_temperature,
                )

            if self.config.generate_audio:
                if self._audio_pipeline is None:
                    voice_synth = build_voice_synthesizer(
                        tts_backend=self.config.tts_backend,
                        speaker_wavs=self.config.speaker_wavs,
                        female_speaker_wavs=self.config.female_speaker_wavs,
                        alternate_voices=self.config.alternate_voices,
                        elevenlabs_api_key=self.config.elevenlabs_api_key,
                        elevenlabs_voice_id=self.config.elevenlabs_voice_id,
                        elevenlabs_female_voice_id=self.config.elevenlabs_female_voice_id,
                        elevenlabs_model_id=self.config.elevenlabs_model_id,
                        elevenlabs_output_format=self.config.elevenlabs_output_format,
                        elevenlabs_language_code=self.config.elevenlabs_language_code,
                        elevenlabs_stability=self.config.elevenlabs_stability,
                        elevenlabs_similarity_boost=self.config.elevenlabs_similarity_boost,
                        elevenlabs_style=self.config.elevenlabs_style,
                        elevenlabs_speed=self.config.elevenlabs_speed,
                        elevenlabs_use_speaker_boost=self.config.elevenlabs_use_speaker_boost,
                        elevenlabs_optimize_streaming_latency=self.config.elevenlabs_optimize_streaming_latency,
                    )
                    self._audio_pipeline = CommentaryAudioPipeline(
                        commentary_generator=self._generator,
                        voice_synthesizer=voice_synth,
                        output_dir=str(self._output_dir) if self._output_dir else None,
                    )

                audio_out = self._build_audio_path(metadata)
                audio_result = self._audio_pipeline.generate_to_file(
                    commentary_event,
                    audio_path=audio_out,
                )
                result["text"] = audio_result.commentary
                result["audio_path"] = str(audio_result.audio_path)
                result["audio_duration_seconds"] = audio_result.audio_duration_seconds
                result["llm_seconds"] = round(audio_result.llm_seconds or 0.0, 3)
                result["tts_seconds"] = round(audio_result.tts_seconds or 0.0, 3)
            elif self.config.generate_text:
                gen_result = self._generator.generate(commentary_event)
                result["text"] = gen_result.commentary

        except Exception as exc:
            result["error"] = str(exc)
            with self._lock:
                self._failed += 1

        with self._lock:
            self._completed.append(result)

        self._write_manifest_entry(result)
        return result

    def _build_audio_path(self, metadata: dict) -> Path:
        action_index = metadata.get("action_index", self._action_index)
        action = metadata.get("event_type_raw", "accion")
        stem = f"{int(action_index):06d}_{_clean_str(action) or 'accion'}"
        suffix = ".wav"
        if self._audio_pipeline is not None:
            synth = self._audio_pipeline.voice_synthesizer
            suffix = str(getattr(synth, "output_suffix", ".wav") or ".wav")
            if not suffix.startswith("."):
                suffix = f".{suffix}"
        audio_dir = self._output_dir / "audio" if self._output_dir else Path("output/commentary/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        return audio_dir / f"{stem}{suffix}"

    def _write_manifest_entry(self, result: dict) -> None:
        if self._manifest_path is None:
            return
        try:
            entry = {
                "action_index": result.get("action_index"),
                "frame_id": result.get("frame_id"),
                "event": result.get("event"),
                "text": result.get("text"),
                "audio_path": result.get("audio_path"),
                "audio_duration_seconds": result.get("audio_duration_seconds"),
                "llm_seconds": result.get("llm_seconds"),
                "tts_seconds": result.get("tts_seconds"),
                "error": result.get("error"),
            }
            with self._manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _collect_completed_jobs(self) -> None:
        with self._lock:
            if self._pending_job is not None and self._pending_job.done():
                try:
                    result = self._pending_job.result()
                    if result and result.get("text"):
                        pass
                except Exception:
                    pass
                finally:
                    self._pending_job = None
            self._maybe_dispatch_next()

    def _drain_queue(self) -> None:
        while True:
            self._collect_completed_jobs()
            if self._pending_job is not None:
                try:
                    self._pending_job.result(timeout=120)
                except Exception:
                    pass
                self._pending_job = None
            self._maybe_dispatch_next()
            if self._pending_job is None and not self._queue:
                break

    def summary(self) -> dict:
        self._drain_queue()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        with self._lock:
            return {
                "total_dispatched": self._action_index,
                "total_completed": len(self._completed),
                "total_failed": self._failed,
                "total_with_audio": sum(1 for r in self._completed if r.get("audio_path")),
                "manifest_path": str(self._manifest_path) if self._manifest_path else None,
            }

    def build_result(self, output_dir: Path | str | None = None) -> dict:
        self.summary()
        out = Path(output_dir).expanduser().resolve() if output_dir else self._output_dir
        if out is None:
            out = Path("output/commentary")
        out.mkdir(parents=True, exist_ok=True)

        events_json_path = out / "commentary_events.json"
        try:
            events_payload = {
                "total": len(self._completed),
                "events": [
                    {"action_index": r.get("action_index"), "event": r.get("event"), "text": r.get("text")}
                    for r in self._completed
                ],
            }
            with events_json_path.open("w", encoding="utf-8") as f:
                json.dump(events_payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {
            "commentary_events_json": str(events_json_path),
            "commentary_manifest_jsonl": str(self._manifest_path) if self._manifest_path else None,
            "total_completed": len(self._completed),
        }

    def close(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._drain_queue()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._pending_job = None
