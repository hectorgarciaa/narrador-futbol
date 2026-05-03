from __future__ import annotations

import base64
from datetime import datetime, timezone
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse

from .generator import CommentaryEvent, OllamaCommentaryGenerator
from .voice import CommentaryAudioPipeline, probe_audio_duration_seconds


DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8788


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


@dataclass(slots=True)
class CommentaryServiceResult:
    status_code: int
    payload: dict[str, Any]


@dataclass(slots=True)
class CommentaryHTTPService:
    commentary_generator: OllamaCommentaryGenerator
    audio_pipeline: CommentaryAudioPipeline | None = None
    default_text_only: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    started_at: float = field(default_factory=time.time)
    last_event_identity: tuple[str, str, str] | None = None
    recent_commentary_by_action: dict[str, str] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.commentary_generator.model

    @property
    def tts_model(self) -> str | None:
        if self.audio_pipeline is None:
            return None
        return self.audio_pipeline.voice_synthesizer.model_name

    @property
    def tts_streaming(self) -> bool:
        if self.audio_pipeline is None:
            return False
        return bool(
            getattr(
                self.audio_pipeline.voice_synthesizer,
                "supports_streaming_audio",
                False,
            )
        )

    def health_payload(self) -> dict[str, Any]:
        backend_name_getter = getattr(
            self.commentary_generator,
            "_backend_display_name",
            None,
        )
        backend_name = (
            str(backend_name_getter()).strip().lower()
            if callable(backend_name_getter)
            else "ollama"
        )
        return {
            "status": "ok",
            "model": self.model,
            "tts_model": self.tts_model,
            "tts_streaming": self.tts_streaming,
            "text_only": self.default_text_only,
            "llm_backend": backend_name,
            "llm_reachable": self.commentary_generator.check_health(),
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }

    def _resolve_request(
        self,
        payload: dict[str, Any],
    ) -> CommentaryServiceResult | tuple[
        CommentaryEvent,
        str | None,
        str | None,
        str | None,
        dict[str, Any],
        bool,
    ]:
        if not isinstance(payload, dict):
            return CommentaryServiceResult(
                status_code=HTTPStatus.BAD_REQUEST,
                payload={"error": "El cuerpo debe ser un objeto JSON."},
            )

        event_payload = payload.get("event", payload)
        audio_out = payload.get("audio_out")
        manifest_path = payload.get("manifest_path")
        requested_mode = str(payload.get("mode") or "").strip().lower() or None
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}
        text_only = bool(payload.get("text_only", self.default_text_only))

        try:
            event = CommentaryEvent.from_dict(event_payload)
        except (TypeError, ValueError) as exc:
            return CommentaryServiceResult(
                status_code=HTTPStatus.BAD_REQUEST,
                payload={"error": str(exc)},
            )
        return (
            event,
            audio_out,
            manifest_path,
            requested_mode,
            metadata or {},
            text_only,
        )

    def _append_manifest(
        self,
        manifest_path: str | None,
        *,
        event: CommentaryEvent,
        requested_mode: str | None,
        metadata: dict[str, Any],
        text_only: bool,
        response_payload: dict[str, Any],
    ) -> None:
        if not manifest_path:
            return

        append_jsonl(
            manifest_path,
            {
                "generated_at_utc": now_iso(),
                "event": event.to_prompt_payload(),
                "mode": requested_mode,
                "metadata": metadata,
                "commentary": response_payload.get("commentary"),
                "audio_path": response_payload.get("audio_path"),
                "model": response_payload.get("model"),
                "tts_model": self.tts_model,
                "text_only": text_only,
                "llm_seconds": response_payload.get("llm_seconds"),
                "tts_seconds": response_payload.get("tts_seconds"),
                "total_seconds": response_payload.get("total_seconds"),
                "audio_duration_seconds": response_payload.get("audio_duration_seconds"),
                "voice_label": response_payload.get("voice_label"),
            },
        )
        response_payload["manifest_path"] = str(Path(manifest_path).expanduser().resolve())

    def _stream_synthesize_to_file(
        self,
        *,
        text: str,
        output_path: Path,
    ):
        if self.audio_pipeline is None:
            raise RuntimeError(
                "El servidor se inicio en modo solo texto y no puede generar audio."
            )
        voice_synthesizer = self.audio_pipeline.voice_synthesizer
        if not bool(getattr(voice_synthesizer, "supports_streaming_audio", False)):
            audio_path_value = voice_synthesizer.synthesize_to_file(text, output_path)
            return audio_path_value

        suffix = str(getattr(voice_synthesizer, "output_suffix", "") or "").strip()
        if suffix and output_path.suffix.lower() != suffix.lower():
            output_path = output_path.with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_mimetype = str(
            getattr(voice_synthesizer, "audio_mimetype", "application/octet-stream")
        )
        wrote_bytes = False
        chunk_index = 0
        with open(output_path, "wb") as f:
            for chunk in voice_synthesizer.iter_audio_chunks(text):
                if not chunk:
                    continue
                f.write(chunk)
                f.flush()
                wrote_bytes = True
                chunk_index += 1
                yield "audio_chunk", {
                    "audio_path": str(output_path),
                    "content_type": audio_mimetype,
                    "chunk_index": chunk_index,
                    "data_base64": base64.b64encode(chunk).decode("ascii"),
                }
        if not wrote_bytes:
            raise RuntimeError("El backend TTS no devolvio audio para la peticion.")
        return output_path

    def _event_identity(self, event: CommentaryEvent) -> tuple[str, str, str]:
        team_scope = event.team_name or event.team_in_favor or ""
        return (
            str(event.action or "").strip().casefold(),
            str(event.player_name or "").strip().casefold(),
            str(team_scope or "").strip().casefold(),
        )

    def _build_duplicate_response(self, event: CommentaryEvent) -> dict[str, Any]:
        team_scope = event.team_name or event.team_in_favor
        payload = {
            "skipped": True,
            "skip_reason": "duplicate_consecutive_event",
            "commentary": None,
            "audio_path": None,
            "model": self.model,
            "event_identity": {
                "action": event.action,
                "player_name": event.player_name,
            },
        }
        if team_scope:
            payload["event_identity"]["team_name"] = team_scope
        return payload

    def _remember_event_identity(self, event: CommentaryEvent) -> None:
        self.last_event_identity = self._event_identity(event)

    def _recent_commentary_for_event(self, event: CommentaryEvent) -> str | None:
        return self.recent_commentary_by_action.get(str(event.action or "").casefold())

    def _remember_commentary(self, event: CommentaryEvent, commentary: str | None) -> None:
        text = str(commentary or "").strip()
        if not text:
            return
        self.recent_commentary_by_action[str(event.action or "").casefold()] = text

    def process_payload(self, payload: dict[str, Any]) -> CommentaryServiceResult:
        resolved = self._resolve_request(payload)
        if isinstance(resolved, CommentaryServiceResult):
            return resolved

        event, audio_out, manifest_path, requested_mode, metadata, text_only = resolved

        try:
            with self.lock:
                if self._event_identity(event) == self.last_event_identity:
                    return CommentaryServiceResult(
                        status_code=HTTPStatus.OK,
                        payload=self._build_duplicate_response(event),
                    )
                if text_only:
                    started = time.perf_counter()
                    result = self.commentary_generator.generate(
                        event,
                        avoid_commentary=self._recent_commentary_for_event(event),
                    )
                    response_payload = {
                        "commentary": result.commentary,
                        "model": result.model,
                        "llm_seconds": round(time.perf_counter() - started, 3),
                    }
                else:
                    if self.audio_pipeline is None:
                        raise RuntimeError(
                            "El servidor se inicio en modo solo texto y no puede generar audio."
                        )
                    result = self.audio_pipeline.generate_to_file(
                        event,
                        audio_path=audio_out,
                        avoid_commentary=self._recent_commentary_for_event(event),
                    )
                    response_payload = {
                        "commentary": result.commentary,
                        "audio_path": str(result.audio_path),
                        "model": result.commentary_result.model,
                        "llm_seconds": round(result.llm_seconds or 0.0, 3),
                        "tts_seconds": round(result.tts_seconds or 0.0, 3),
                        "total_seconds": round(result.total_seconds or 0.0, 3),
                        "audio_duration_seconds": (
                            round(result.audio_duration_seconds, 3)
                            if result.audio_duration_seconds is not None
                            else None
                        ),
                        "voice_label": result.voice_label,
                    }
                self._remember_event_identity(event)
                self._remember_commentary(event, response_payload.get("commentary"))
        except Exception as exc:
            return CommentaryServiceResult(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                payload={"error": str(exc)},
            )

        self._append_manifest(
            manifest_path,
            event=event,
            requested_mode=requested_mode,
            metadata=metadata,
            text_only=text_only,
            response_payload=response_payload,
        )

        return CommentaryServiceResult(
            status_code=HTTPStatus.OK,
            payload=response_payload,
        )

    def process_payload_stream(self, payload: dict[str, Any]):
        resolved = self._resolve_request(payload)
        if isinstance(resolved, CommentaryServiceResult):
            return resolved

        event, audio_out, manifest_path, requested_mode, metadata, text_only = resolved

        def _event_stream():
            total_start = time.perf_counter()
            yield "accepted", {
                "generated_at_utc": now_iso(),
                "text_only": text_only,
                "mode": requested_mode,
            }
            try:
                duplicate_payload: dict[str, Any] | None = None
                with self.lock:
                    if self._event_identity(event) == self.last_event_identity:
                        duplicate_payload = self._build_duplicate_response(event)
                    else:
                        llm_start = time.perf_counter()
                        commentary_result = self.commentary_generator.generate(
                            event,
                            avoid_commentary=self._recent_commentary_for_event(event),
                        )
                        llm_seconds = time.perf_counter() - llm_start
                        yield "commentary", {
                            "commentary": commentary_result.commentary,
                            "model": commentary_result.model,
                            "llm_seconds": round(llm_seconds, 3),
                        }

                        if text_only:
                            response_payload = {
                                "commentary": commentary_result.commentary,
                                "model": commentary_result.model,
                                "llm_seconds": round(llm_seconds, 3),
                            }
                        else:
                            if self.audio_pipeline is None:
                                raise RuntimeError(
                                    "El servidor se inicio en modo solo texto y no puede generar audio."
                                )
                            output_path = (
                                Path(audio_out).expanduser().resolve()
                                if audio_out is not None
                                else self.audio_pipeline.build_output_path(
                                    commentary_result.event
                                )
                            )
                            yield "tts_start", {
                                "audio_path": str(output_path),
                                "streaming_audio": bool(
                                    getattr(
                                        self.audio_pipeline.voice_synthesizer,
                                        "supports_streaming_audio",
                                        False,
                                    )
                                ),
                            }
                            tts_start = time.perf_counter()
                            audio_path_value = yield from self._stream_synthesize_to_file(
                                text=commentary_result.commentary,
                                output_path=output_path,
                            )
                            if audio_path_value is None:
                                raise RuntimeError(
                                    "No se pudo resolver la ruta del audio streaming."
                                )
                            voice_label = getattr(
                                self.audio_pipeline.voice_synthesizer,
                                "last_voice_label",
                                None,
                            )
                            audio_duration_seconds = probe_audio_duration_seconds(
                                audio_path_value
                            )
                            tts_seconds = time.perf_counter() - tts_start
                            response_payload = {
                                "commentary": commentary_result.commentary,
                                "audio_path": str(audio_path_value),
                                "model": commentary_result.model,
                                "llm_seconds": round(llm_seconds, 3),
                                "tts_seconds": round(tts_seconds, 3),
                                "total_seconds": round(
                                    time.perf_counter() - total_start,
                                    3,
                                ),
                                "audio_duration_seconds": (
                                    round(audio_duration_seconds, 3)
                                    if audio_duration_seconds is not None
                                    else None
                                ),
                                "voice_label": voice_label,
                            }
                        self._remember_event_identity(event)
                        self._remember_commentary(event, response_payload.get("commentary"))

                if duplicate_payload is not None:
                    yield "skipped", duplicate_payload
                    yield "completed", duplicate_payload
                    return

                self._append_manifest(
                    manifest_path,
                    event=event,
                    requested_mode=requested_mode,
                    metadata=metadata,
                    text_only=text_only,
                    response_payload=response_payload,
                )
                yield "completed", response_payload
            except Exception as exc:
                yield "error", {"error": str(exc)}

        return _event_stream()


class CommentaryHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: CommentaryHTTPService,
    ) -> None:
        self.service = service
        super().__init__(server_address, CommentaryRequestHandler)


class CommentaryRequestHandler(BaseHTTPRequestHandler):
    server_version = "CommentariesHTTP/1.0"

    def _write_json(self, status_code: int, payload: dict[str, Any], send_body: bool) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status_code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _write_sse_event(self, event_name: str, payload: dict[str, Any]) -> None:
        body = (
            f"event: {event_name}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        ).encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _route(self) -> str:
        return urlparse(self.path).path

    def _handle_health(self, send_body: bool) -> None:
        self._write_json(
            HTTPStatus.OK,
            self.server.service.health_payload(),
            send_body=send_body,
        )

    def _handle_not_found(self, send_body: bool) -> None:
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": f"Ruta no encontrada: {self._route()}"},
            send_body=send_body,
        )

    def do_GET(self) -> None:
        if self._route() in {"/", "/health"}:
            self._handle_health(send_body=True)
            return
        self._handle_not_found(send_body=True)

    def do_HEAD(self) -> None:
        if self._route() in {"/", "/health"}:
            self._handle_health(send_body=False)
            return
        self._handle_not_found(send_body=False)

    def do_POST(self) -> None:
        route = self._route()
        if route not in {"/api/commentaries", "/api/commentaries/stream"}:
            self._handle_not_found(send_body=True)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw_body = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"JSON invalido: {exc.msg}"},
                send_body=True,
            )
            return

        if route == "/api/commentaries/stream":
            stream_response = self.server.service.process_payload_stream(payload)
            if isinstance(stream_response, CommentaryServiceResult):
                self._write_json(
                    stream_response.status_code,
                    stream_response.payload,
                    send_body=True,
                )
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for event_name, event_payload in stream_response:
                self._write_sse_event(event_name, event_payload)
            return

        result = self.server.service.process_payload(payload)
        self._write_json(result.status_code, result.payload, send_body=True)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_http_service(
    commentary_generator: OllamaCommentaryGenerator,
    audio_pipeline: CommentaryAudioPipeline | None = None,
    *,
    text_only: bool = False,
) -> CommentaryHTTPService:
    return CommentaryHTTPService(
        commentary_generator=commentary_generator,
        audio_pipeline=audio_pipeline,
        default_text_only=bool(text_only),
    )


def create_http_server(
    commentary_generator: OllamaCommentaryGenerator,
    audio_pipeline: CommentaryAudioPipeline | None = None,
    *,
    host: str = DEFAULT_SERVER_HOST,
    port: int = DEFAULT_SERVER_PORT,
    text_only: bool = False,
) -> CommentaryHTTPServer:
    service = build_http_service(
        commentary_generator=commentary_generator,
        audio_pipeline=audio_pipeline,
        text_only=text_only,
    )
    return CommentaryHTTPServer((host, int(port)), service)
