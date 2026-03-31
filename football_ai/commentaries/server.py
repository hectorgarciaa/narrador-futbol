from __future__ import annotations

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
from .voice import CommentaryAudioPipeline


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

    @property
    def model(self) -> str:
        return self.commentary_generator.model

    @property
    def tts_model(self) -> str | None:
        if self.audio_pipeline is None:
            return None
        return self.audio_pipeline.voice_synthesizer.model_name

    def health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model,
            "tts_model": self.tts_model,
            "text_only": self.default_text_only,
            "ollama_reachable": self.commentary_generator.check_health(),
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }

    def process_payload(self, payload: dict[str, Any]) -> CommentaryServiceResult:
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

        try:
            with self.lock:
                if text_only:
                    started = time.perf_counter()
                    result = self.commentary_generator.generate(event)
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
                    )
                    response_payload = {
                        "commentary": result.commentary,
                        "audio_path": str(result.audio_path),
                        "model": result.commentary_result.model,
                        "llm_seconds": round(result.llm_seconds or 0.0, 3),
                        "tts_seconds": round(result.tts_seconds or 0.0, 3),
                        "total_seconds": round(result.total_seconds or 0.0, 3),
                    }
        except Exception as exc:
            return CommentaryServiceResult(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                payload={"error": str(exc)},
            )

        if manifest_path:
            append_jsonl(
                manifest_path,
                {
                    "generated_at_utc": now_iso(),
                    "event": event.to_prompt_payload(),
                    "mode": requested_mode,
                    "metadata": metadata or {},
                    "commentary": response_payload.get("commentary"),
                    "audio_path": response_payload.get("audio_path"),
                    "model": response_payload.get("model"),
                    "tts_model": self.tts_model,
                    "text_only": text_only,
                    "llm_seconds": response_payload.get("llm_seconds"),
                    "tts_seconds": response_payload.get("tts_seconds"),
                    "total_seconds": response_payload.get("total_seconds"),
                },
            )
            response_payload["manifest_path"] = str(
                Path(manifest_path).expanduser().resolve()
            )

        return CommentaryServiceResult(
            status_code=HTTPStatus.OK,
            payload=response_payload,
        )


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
        if route != "/api/commentaries":
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
