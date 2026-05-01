from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from football_ai.commentaries.voice import probe_audio_duration_seconds
from football_ai.actions import (
    PathCRFInferenceConfig,
    PathCRFRenderConfig,
    run_pathcrf_pipeline,
)
from football_ai.actions.pathcrf_adapter import PathCRFAdapterConfig
from football_ai.core import convert_to_serializable


logger = logging.getLogger(__name__)

AUDIO_TIMELINE_EPSILON_S = 0.05
CONTEXT_COMMENT_PROBABILITY = 0.3
CONTEXT_COMMENT_COOLDOWN_S = 35.0
URGENT_COMMENTARY_ACTIONS = {"tiro", "gol"}

ENV_ENABLE = "NARRADOR_APP_ENABLE_LIVE_COMMENTARY"
ENV_SERVICE_URL = "NARRADOR_APP_COMMENTARY_SERVICE_URL"
ENV_MANIFEST_PATH = "NARRADOR_APP_COMMENTARY_MANIFEST_PATH"
ENV_AUDIO_DIR = "NARRADOR_APP_COMMENTARY_AUDIO_DIR"
ENV_MODE = "NARRADOR_APP_COMMENTARY_MODE"
ENV_RUN_DIR = "NARRADOR_APP_RUN_DIR"
ENV_RUN_ID = "NARRADOR_APP_RUN_ID"
ENV_PATHCRF_REPO_PATH = "NARRADOR_APP_PATHCRF_REPO_PATH"
ENV_PATHCRF_TRIAL = "NARRADOR_APP_PATHCRF_TRIAL"
ENV_PATHCRF_DEVICE = "NARRADOR_APP_PATHCRF_DEVICE"
ENV_PATHCRF_INTERVAL_FRAMES = "NARRADOR_APP_PATHCRF_INTERVAL_FRAMES"
ENV_PATHCRF_MIN_FRAMES = "NARRADOR_APP_PATHCRF_MIN_FRAMES"
ENV_PATHCRF_STABILIZATION_LAG_S = "NARRADOR_APP_PATHCRF_STABILIZATION_LAG_S"
ENV_PATHCRF_FPS = "NARRADOR_APP_PATHCRF_FPS"


@dataclass(slots=True)
class LiveCommentaryBridgeConfig:
    service_url: str
    manifest_path: Path
    audio_dir: Path
    run_dir: Path
    run_id: str
    commentary_mode: str = "deferred"
    repo_path: Path = Path("football_ai/actions/repo/pathcrf")
    trial: int = 120
    device: str = "auto"
    snapshot_interval_frames: int = 75
    min_frames: int = 50
    stabilization_lag_s: float = 3.0
    fps: float = 25.0


@dataclass(slots=True)
class SnapshotTask:
    frame_id: int
    observed_seconds: float
    tracks_snapshot: dict[str, Any]


class CompositeFrameHook:
    def __init__(self, hooks: list[Callable[[dict[str, Any], int], None]]) -> None:
        self.hooks = [hook for hook in hooks if callable(hook)]
        self.role_session = self._infer_role_session()

    def _infer_role_session(self) -> Any | None:
        for hook in self.hooks:
            owner = getattr(hook, "__self__", None)
            role_session = getattr(owner, "role_session", None)
            if role_session is not None:
                return role_session
        return None

    def __call__(self, tracks: dict[str, Any], n_frame: int) -> None:
        for hook in self.hooks:
            hook(tracks, n_frame)


class AppLiveCommentaryBridge:
    def __init__(self, config: LiveCommentaryBridgeConfig) -> None:
        self.config = config
        self.audio_dir = config.audio_dir.expanduser().resolve()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = config.run_dir.expanduser().resolve()
        self.pathcrf_root = self.run_dir / "pathcrf_live"
        self.pathcrf_root.mkdir(parents=True, exist_ok=True)
        self.latest_snapshot_path = self.pathcrf_root / "latest_tracks_snapshot.json"
        self.latest_output_dir = self.pathcrf_root / "latest"
        self.final_output_dir = self.pathcrf_root / "final"
        self._condition = threading.Condition()
        self._pending_task: SnapshotTask | None = None
        self._stop_requested = False
        self._last_submitted_frame = -1
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._dispatched_signatures: set[str] = set()
        self._dispatched_semantic_event_times: dict[
            tuple[str, str, str, str],
            list[float],
        ] = {}
        self._intro_audio_duration_seconds = self._initial_intro_audio_duration()
        self._audio_busy_until_wall_time = self._initial_audio_busy_until(
            self._intro_audio_duration_seconds
        )
        self._audio_busy_until_event_time_s = max(
            0.0,
            self._intro_audio_duration_seconds,
        )
        self._current_audio_interruptible = False
        self._last_context_event_time_s = -1_000_000.0
        self._context_rng = random.Random(self._stable_seed(config.run_id))
        self._lineup_context = self._load_lineup_context()
        self._worker.start()

    def _initial_intro_audio_duration(self) -> float:
        candidates = [self.audio_dir / "000_intro.wav"]
        candidates.extend(sorted(self.audio_dir.glob("000_intro.*")))
        seen: set[Path] = set()
        for intro_audio_path in candidates:
            resolved_path = intro_audio_path.expanduser().resolve()
            if resolved_path in seen or not resolved_path.exists():
                continue
            seen.add(resolved_path)
            intro_duration_seconds = probe_audio_duration_seconds(resolved_path)
            if intro_duration_seconds is not None and intro_duration_seconds > 0.0:
                return float(intro_duration_seconds)
        return 0.0

    def _initial_audio_busy_until(self, intro_duration_seconds: float) -> float:
        if intro_duration_seconds <= 0.0:
            return 0.0
        return time.perf_counter() + float(intro_duration_seconds)

    def on_frame(self, tracks: dict[str, Any], n_frame: int) -> None:
        if (n_frame + 1) < int(self.config.min_frames):
            return
        if self._last_submitted_frame >= 0:
            delta = int(n_frame) - int(self._last_submitted_frame)
            if delta < int(self.config.snapshot_interval_frames):
                return

        snapshot = copy.deepcopy(tracks)
        task = SnapshotTask(
            frame_id=int(n_frame),
            observed_seconds=float(n_frame + 1) / max(float(self.config.fps), 1e-6),
            tracks_snapshot=snapshot,
        )
        with self._condition:
            self._pending_task = task
            self._last_submitted_frame = int(n_frame)
            self._condition.notify_all()

    def finalize(self, final_tracks_path: str | Path | None) -> None:
        self.close()
        if final_tracks_path is None:
            return
        try:
            self._process_tracks_file(
                tracks_path=Path(final_tracks_path).expanduser().resolve(),
                output_dir=self.final_output_dir,
                observed_seconds=None,
                final_pass=True,
            )
        except Exception:
            logger.exception("No se pudo ejecutar el flush final de PathCRF live.")

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._pending_task = None
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_task is None and not self._stop_requested:
                    self._condition.wait(timeout=0.5)
                if self._stop_requested:
                    return
                task = self._pending_task
                self._pending_task = None
            if task is None:
                continue
            try:
                self._write_tracks_snapshot(task.tracks_snapshot, self.latest_snapshot_path)
                self._process_tracks_file(
                    tracks_path=self.latest_snapshot_path,
                    output_dir=self.latest_output_dir,
                    observed_seconds=float(task.observed_seconds),
                    final_pass=False,
                )
            except Exception:
                logger.exception(
                    "Fallo en la ejecución incremental de PathCRF live para el frame %s.",
                    task.frame_id,
                )

    def _write_tracks_snapshot(self, tracks: dict[str, Any], target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(
                convert_to_serializable(tracks),
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )

    @staticmethod
    def _stable_seed(value: str) -> int:
        digest = hashlib.sha1(str(value or "run").encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _load_lineup_context(self) -> dict[str, Any]:
        lineup_path = self.run_dir / "lineup_spec.json"
        if not lineup_path.exists():
            return {"teams": []}
        try:
            with lineup_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("No se pudo leer lineup_spec.json para comentarios de contexto.")
            return {"teams": []}
        if not isinstance(payload, dict):
            return {"teams": []}
        teams = payload.get("teams")
        if not isinstance(teams, list):
            teams = []
        return {"teams": [team for team in teams if isinstance(team, dict)]}

    def _lineup_team(self, team_name: str | None) -> dict[str, Any] | None:
        wanted = str(team_name or "").strip().casefold()
        if not wanted:
            return None
        for team in self._lineup_context.get("teams") or []:
            candidate = str(team.get("team_name") or "").strip().casefold()
            if candidate == wanted:
                return dict(team)
        return None

    def _players_for_slots(
        self,
        team: dict[str, Any] | None,
        slot_prefixes: tuple[str, ...],
        *,
        limit: int = 3,
    ) -> list[str]:
        if not team:
            return []
        players_by_slot = team.get("players_by_slot") or {}
        if not isinstance(players_by_slot, dict):
            return []
        selected: list[str] = []
        for slot, player in players_by_slot.items():
            slot_key = str(slot or "").strip().upper()
            player_name = str(player or "").strip()
            if not player_name:
                continue
            if any(slot_key.startswith(prefix) for prefix in slot_prefixes):
                selected.append(player_name)
        return selected[: int(limit)]

    def _simulated_standings_context(
        self,
        team_name: str | None,
        opponent_team_name: str | None,
    ) -> str | None:
        team = str(team_name or "").strip()
        opponent = str(opponent_team_name or "").strip()
        if not team or not opponent:
            return None
        digest = hashlib.sha1(
            f"{self.config.run_id}:{team}:{opponent}".encode("utf-8")
        ).hexdigest()
        team_rank = (int(digest[:2], 16) % 8) + 1
        opponent_rank = (int(digest[2:4], 16) % 8) + 1
        if opponent_rank == team_rank:
            opponent_rank = (opponent_rank % 8) + 1
        team_points = max(22, 62 - (team_rank * 3) + (int(digest[4:6], 16) % 5))
        opponent_points = max(
            22,
            62 - (opponent_rank * 3) + (int(digest[6:8], 16) % 5),
        )
        return (
            f"{team} llega {team_rank} en la tabla con {team_points} puntos; "
            f"{opponent} aparece {opponent_rank} con {opponent_points} puntos"
        )

    def _tactical_context(self, team_name: str | None) -> str | None:
        team = self._lineup_team(team_name)
        if not team:
            return None
        team_label = str(team.get("team_name") or team_name or "").strip()
        center_backs = self._players_for_slots(team, ("DFC",), limit=3)
        midfielders = self._players_for_slots(team, ("MC", "MCD", "MCO"), limit=3)
        forwards = self._players_for_slots(team, ("DC", "EI", "ED"), limit=3)
        if len(center_backs) >= 2:
            return (
                f"defensa de {team_label}: centrales "
                f"{' y '.join(center_backs[:2])}"
            )
        if len(midfielders) >= 2:
            return (
                f"centro del campo de {team_label}: "
                f"{', '.join(midfielders[:3])}"
            )
        if len(forwards) >= 2:
            return f"ataque de {team_label}: {' y '.join(forwards[:2])}"
        return None

    def _build_context_event(
        self,
        event_payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        team_name = event_payload.get("team_name")
        opponent_team_name = event_payload.get("opponent_team_name")
        event_time_s = event_payload.get("event_time_s")
        tactical_context = self._tactical_context(team_name)
        standings_context = self._simulated_standings_context(
            team_name,
            opponent_team_name,
        )
        if not tactical_context and not standings_context:
            return None
        return {
            "action": "contexto",
            "event_time_s": event_time_s,
            "team_name": team_name,
            "opponent_team_name": opponent_team_name,
            "field_zone": event_payload.get("field_zone"),
            "play_context": tactical_context,
            "match_score": standings_context,
            "action_index": event_payload.get("action_index"),
        }

    def _should_emit_context_comment(
        self,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
        *,
        event_time_s: float,
        final_pass: bool,
    ) -> bool:
        if final_pass:
            return False
        if str(metadata.get("field_zone_key") or "").strip() != "iniciacion":
            return False
        action = str(event_payload.get("action") or "").strip().casefold()
        if action in URGENT_COMMENTARY_ACTIONS:
            return False
        if event_time_s - self._last_context_event_time_s < CONTEXT_COMMENT_COOLDOWN_S:
            return False
        return self._context_rng.random() <= CONTEXT_COMMENT_PROBABILITY

    def _is_urgent_event(
        self,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> bool:
        del metadata
        action = str(event_payload.get("action") or "").strip().casefold()
        return action in URGENT_COMMENTARY_ACTIONS

    def _should_interrupt_current_audio(
        self,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
        *,
        request_started: float,
    ) -> bool:
        return (
            self._current_audio_interruptible
            and request_started < self._audio_busy_until_wall_time
            and self._is_urgent_event(event_payload, metadata)
        )

    def _update_audio_busy_after_response(
        self,
        *,
        request_started: float,
        event_time_s: float,
        response_payload: dict[str, Any],
        interrupting: bool,
        interruptible_audio: bool,
    ) -> None:
        try:
            total_seconds = float(response_payload.get("total_seconds") or 0.0)
            audio_duration_seconds = float(
                response_payload.get("audio_duration_seconds") or 0.0
            )
        except (TypeError, ValueError):
            total_seconds = 0.0
            audio_duration_seconds = 0.0

        wall_end = request_started + total_seconds + audio_duration_seconds
        event_end = event_time_s + total_seconds + audio_duration_seconds
        if interrupting:
            self._audio_busy_until_wall_time = wall_end
            self._audio_busy_until_event_time_s = event_end
        else:
            self._audio_busy_until_wall_time = max(
                self._audio_busy_until_wall_time,
                wall_end,
            )
            self._audio_busy_until_event_time_s = max(
                self._audio_busy_until_event_time_s,
                event_end,
            )
        self._current_audio_interruptible = bool(interruptible_audio)

    def _process_tracks_file(
        self,
        *,
        tracks_path: Path,
        output_dir: Path,
        observed_seconds: float | None,
        final_pass: bool,
    ) -> None:
        inference_config = PathCRFInferenceConfig(
            repo_path=self.config.repo_path,
            trial=int(self.config.trial),
            device=self.config.device,
        )
        render_config = PathCRFRenderConfig(enabled=False)
        adapter_config = PathCRFAdapterConfig(fps=float(self.config.fps))
        result = run_pathcrf_pipeline(
            output_dir=output_dir,
            tracks_path=tracks_path,
            adapter_config=adapter_config,
            inference_config=inference_config,
            render_config=render_config,
        )
        if result.commentary_json_path is None or not result.commentary_json_path.exists():
            return
        self._dispatch_new_events(
            commentary_json_path=result.commentary_json_path,
            observed_seconds=observed_seconds,
            final_pass=final_pass,
        )

    def _dispatch_new_events(
        self,
        *,
        commentary_json_path: Path,
        observed_seconds: float | None,
        final_pass: bool,
    ) -> None:
        with commentary_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        commentary_events = list(payload.get("commentary_events") or [])
        if not commentary_events:
            return

        stabilization_limit = None
        if not final_pass and observed_seconds is not None:
            stabilization_limit = max(0.0, float(observed_seconds) - float(self.config.stabilization_lag_s))

        for item in commentary_events:
            event_payload = dict(item.get("event") or {})
            metadata = dict(item.get("metadata") or {})
            try:
                event_time_s = float(event_payload.get("event_time_s"))
            except (TypeError, ValueError):
                continue
            if stabilization_limit is not None and event_time_s > stabilization_limit:
                continue

            signature = self._event_signature(event_payload, metadata)
            if signature in self._dispatched_signatures:
                continue
            if self._is_near_duplicate_semantic_event(event_payload, metadata):
                continue

            context_audio_generated = self._maybe_dispatch_context_comment(
                event_payload=event_payload,
                metadata=metadata,
                signature=signature,
                event_time_s=event_time_s,
                final_pass=final_pass,
            )
            request_started = time.perf_counter()
            interrupt_current_audio = self._should_interrupt_current_audio(
                event_payload,
                metadata,
                request_started=request_started,
            )
            event_for_request = dict(event_payload)
            if interrupt_current_audio:
                event_for_request["intensity"] = "interrupcion"
                interrupt_hint = (
                    "hay que cortar un comentario de contexto porque aparece una accion peligrosa"
                )
                existing_context = str(event_for_request.get("play_context") or "").strip()
                event_for_request["play_context"] = (
                    f"{existing_context}. {interrupt_hint}"
                    if existing_context
                    else interrupt_hint
                )

            text_only, text_only_reason = self._audio_policy_for_event(
                event_time_s=event_time_s,
                request_started=request_started,
                allow_interrupt=interrupt_current_audio,
            )
            if context_audio_generated:
                text_only = True
                text_only_reason = "context_commentary_replaced_initiation_action"
            audio_out = self.audio_dir / self._audio_filename_for_event(
                event_payload=event_for_request,
                metadata=metadata,
                signature=signature,
            )
            request_metadata = {
                **metadata,
                "run_id": self.config.run_id,
                "source": "pathcrf_live",
                "event_signature": signature,
                "final_pass": bool(final_pass),
            }
            if interrupt_current_audio:
                request_metadata["interrupt_audio"] = True
                request_metadata["interrupt_reason"] = (
                    "urgent_action_interrupts_context_commentary"
                )
            if text_only_reason is not None:
                request_metadata["text_only_reason"] = text_only_reason
                request_metadata["audio_busy_until_event_time_s"] = round(
                    self._audio_busy_until_event_time_s,
                    3,
                )
            response_status, response_payload = self._post_commentary_request(
                {
                    "event": event_for_request,
                    "audio_out": str(audio_out),
                    "manifest_path": str(self.config.manifest_path),
                    "mode": self.config.commentary_mode,
                    "metadata": request_metadata,
                    "text_only": bool(text_only),
                }
            )
            if response_status != int(HTTPStatus.OK):
                logger.warning(
                    "El servidor de comentarios rechazo un evento PathCRF live: %s",
                    response_payload.get("error") or response_payload,
                )
                continue

            self._dispatched_signatures.add(signature)
            self._remember_semantic_event(event_payload, metadata)
            if bool(response_payload.get("skipped")):
                logger.info(
                    "Evento live omitido por duplicado consecutivo: %s (%s).",
                    event_for_request.get("action"),
                    event_for_request.get("player_name"),
                )
                continue
            if not text_only and response_payload.get("audio_path"):
                self._update_audio_busy_after_response(
                    request_started=request_started,
                    event_time_s=event_time_s,
                    response_payload=response_payload,
                    interrupting=interrupt_current_audio,
                    interruptible_audio=False,
                )
            logger.info(
                "Comentario live generado para %s en %.2fs%s.",
                event_for_request.get("action"),
                event_time_s,
                f" (solo texto: {text_only_reason})" if text_only else "",
            )

    def _audio_policy_for_event(
        self,
        *,
        event_time_s: float,
        request_started: float,
        allow_interrupt: bool = False,
    ) -> tuple[bool, str | None]:
        if allow_interrupt:
            return False, None
        if request_started < self._audio_busy_until_wall_time:
            return True, "audio_slot_busy"
        if (
            event_time_s + AUDIO_TIMELINE_EPSILON_S
            < self._audio_busy_until_event_time_s
        ):
            return True, "audio_event_window_elapsed"
        return False, None

    def _maybe_dispatch_context_comment(
        self,
        *,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
        signature: str,
        event_time_s: float,
        final_pass: bool,
    ) -> bool:
        if not self._should_emit_context_comment(
            event_payload,
            metadata,
            event_time_s=event_time_s,
            final_pass=final_pass,
        ):
            return False

        context_event = self._build_context_event(event_payload)
        if context_event is None:
            return False

        request_started = time.perf_counter()
        text_only, _ = self._audio_policy_for_event(
            event_time_s=event_time_s,
            request_started=request_started,
        )
        if text_only:
            return False

        context_signature = hashlib.sha1(
            f"context:{signature}".encode("utf-8")
        ).hexdigest()
        if context_signature in self._dispatched_signatures:
            return False

        audio_out = self.audio_dir / self._audio_filename_for_event(
            event_payload=context_event,
            metadata=metadata,
            signature=context_signature,
        )
        request_metadata = {
            **metadata,
            "run_id": self.config.run_id,
            "source": "pathcrf_live",
            "event_signature": context_signature,
            "base_event_signature": signature,
            "final_pass": bool(final_pass),
            "event_kind": "context",
            "commentary_priority": "context",
            "context_trigger_action": event_payload.get("action"),
            "interruptible_audio": True,
        }
        response_status, response_payload = self._post_commentary_request(
            {
                "event": context_event,
                "audio_out": str(audio_out),
                "manifest_path": str(self.config.manifest_path),
                "mode": self.config.commentary_mode,
                "metadata": request_metadata,
                "text_only": False,
            }
        )
        if response_status != int(HTTPStatus.OK):
            logger.warning(
                "El servidor rechazo un comentario de contexto live: %s",
                response_payload.get("error") or response_payload,
            )
            return False

        self._dispatched_signatures.add(context_signature)
        if bool(response_payload.get("skipped")):
            return False
        if response_payload.get("audio_path"):
            self._update_audio_busy_after_response(
                request_started=request_started,
                event_time_s=event_time_s,
                response_payload=response_payload,
                interrupting=False,
                interruptible_audio=True,
            )
            self._last_context_event_time_s = float(event_time_s)
            logger.info(
                "Comentario de contexto live generado en %.2fs.",
                event_time_s,
            )
            return True
        return False

    def _post_commentary_request(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            f"{self.config.service_url.rstrip('/')}/api/commentaries",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=300.0) as response:
                return int(response.status), json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = {"error": text or str(exc)}
            return int(exc.code), parsed
        except Exception as exc:
            return int(HTTPStatus.INTERNAL_SERVER_ERROR), {"error": str(exc)}

    def _event_signature(self, event_payload: dict[str, Any], metadata: dict[str, Any]) -> str:
        signature_payload = {
            "action": event_payload.get("action"),
            "event_time_centis": int(round(float(event_payload.get("event_time_s") or 0.0) * 100.0)),
            "track_id": metadata.get("track_id"),
            "receiver_track_id": metadata.get("receiver_track_id"),
            "pathcrf_player_id": metadata.get("pathcrf_player_id"),
            "event_type_semantic": metadata.get("event_type_semantic"),
        }
        raw = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _semantic_event_key(
        self,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        return (
            str(event_payload.get("action") or "").strip().casefold(),
            str(metadata.get("track_id") or "").strip(),
            str(metadata.get("receiver_track_id") or "").strip(),
            str(metadata.get("event_type_semantic") or "").strip().casefold(),
        )

    def _is_near_duplicate_semantic_event(
        self,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
        *,
        tolerance_s: float = 2.0,
    ) -> bool:
        key = self._semantic_event_key(event_payload, metadata)
        try:
            event_time_s = float(event_payload.get("event_time_s") or 0.0)
        except (TypeError, ValueError):
            event_time_s = 0.0
        return any(
            abs(float(previous_time_s) - event_time_s) <= float(tolerance_s)
            for previous_time_s in self._dispatched_semantic_event_times.get(key, [])
        )

    def _remember_semantic_event(
        self,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        key = self._semantic_event_key(event_payload, metadata)
        try:
            event_time_s = float(event_payload.get("event_time_s") or 0.0)
        except (TypeError, ValueError):
            event_time_s = 0.0
        self._dispatched_semantic_event_times.setdefault(key, []).append(event_time_s)

    def _audio_filename_for_event(
        self,
        *,
        event_payload: dict[str, Any],
        metadata: dict[str, Any],
        signature: str,
    ) -> str:
        event_time_ms = int(round(float(event_payload.get("event_time_s") or 0.0) * 1000.0))
        action_slug = self._slugify(str(event_payload.get("action") or "accion"))
        track_suffix = metadata.get("track_id")
        track_slug = self._slugify(str(track_suffix)) if track_suffix is not None else "na"
        return f"{event_time_ms:08d}_{action_slug}_{track_slug}_{signature[:10]}.wav"

    @staticmethod
    def _slugify(value: str) -> str:
        text = re.sub(r"\s+", "_", str(value).strip().lower())
        text = re.sub(r"[^a-z0-9._-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("._-")
        return text or "item"


def create_app_live_commentary_bridge_from_env() -> AppLiveCommentaryBridge | None:
    enabled = str(os.environ.get(ENV_ENABLE) or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    service_url = str(os.environ.get(ENV_SERVICE_URL) or "").strip()
    manifest_path = str(os.environ.get(ENV_MANIFEST_PATH) or "").strip()
    audio_dir = str(os.environ.get(ENV_AUDIO_DIR) or "").strip()
    run_dir = str(os.environ.get(ENV_RUN_DIR) or "").strip()
    run_id = str(os.environ.get(ENV_RUN_ID) or "").strip() or "app"
    if not service_url or not manifest_path or not audio_dir or not run_dir:
        logger.warning(
            "Live commentary bridge deshabilitado: faltan variables de entorno obligatorias."
        )
        return None

    return AppLiveCommentaryBridge(
        LiveCommentaryBridgeConfig(
            service_url=service_url,
            manifest_path=Path(manifest_path),
            audio_dir=Path(audio_dir),
            run_dir=Path(run_dir),
            run_id=run_id,
            commentary_mode=str(os.environ.get(ENV_MODE) or "deferred").strip().lower() or "deferred",
            repo_path=Path(os.environ.get(ENV_PATHCRF_REPO_PATH) or "football_ai/actions/repo/pathcrf"),
            trial=int(os.environ.get(ENV_PATHCRF_TRIAL) or 120),
            device=str(os.environ.get(ENV_PATHCRF_DEVICE) or "auto").strip() or "auto",
            snapshot_interval_frames=max(1, int(os.environ.get(ENV_PATHCRF_INTERVAL_FRAMES) or 75)),
            min_frames=max(1, int(os.environ.get(ENV_PATHCRF_MIN_FRAMES) or 50)),
            stabilization_lag_s=max(
                0.0,
                float(os.environ.get(ENV_PATHCRF_STABILIZATION_LAG_S) or 3.0),
            ),
            fps=max(1.0, float(os.environ.get(ENV_PATHCRF_FPS) or 25.0)),
        )
    )


def compose_frame_hooks(*hooks: Callable[[dict[str, Any], int], None] | None) -> Callable[[dict[str, Any], int], None] | None:
    valid_hooks = [hook for hook in hooks if callable(hook)]
    if not valid_hooks:
        return None
    if len(valid_hooks) == 1:
        return valid_hooks[0]
    return CompositeFrameHook(valid_hooks)
