#!/usr/bin/env python3
"""
Interfaz web ligera para introducir alineaciones y lanzar `scripts/track.py`.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import get_config
from football_ai.positions import (
    LineupSpecError,
    get_formation_catalog,
    sanitize_video_stem,
    validate_lineup_payload,
)


STATIC_ROOT = Path(__file__).resolve().parent / "static"
RUNS_ROOT = PROJECT_ROOT / "output" / "interfaz" / "runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

RUNS = {}
RUNS_LOCK = threading.Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_run_paths(run_id):
    run_dir = RUNS_ROOT / str(run_id)
    return {
        "run_dir": run_dir,
        "status_path": run_dir / "status.json",
        "spec_path": run_dir / "lineup_spec.json",
        "log_path": run_dir / "track.log",
    }


def config_video_shortcuts():
    config = get_config()
    data_paths = config.paths.get("data", {})
    shortcuts = []
    for key in sorted(data_paths.keys()):
        if data_paths.get(key) is None:
            continue
        resolved_path = config.get_path("paths", "data", key)
        if resolved_path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
            continue
        shortcuts.append(
            {
                "key": str(key),
                "path": str(resolved_path),
            }
        )
    return shortcuts


def initial_status_payload(run_id, lineup_spec, run_paths):
    teams = lineup_spec["teams"]
    video_source = str(lineup_spec.get("video_source") or "").strip()
    video_stem = sanitize_video_stem(video_source) if video_source else None
    return {
        "run_id": str(run_id),
        "status": "queued",
        "created_at_utc": now_iso(),
        "updated_at_utc": now_iso(),
        "video_source": video_source,
        "video_stem_hint": video_stem,
        "teams": [
            {
                "team_name": str(team["team_name"]),
                "team_color": str(team["team_color"]),
                "formation": str(team["formation"]),
            }
            for team in teams
        ],
        "spec_path": str(run_paths["spec_path"]),
        "log_path": str(run_paths["log_path"]),
        "run_dir": str(run_paths["run_dir"]),
    }


def update_status_file(run_id, **changes):
    run_paths = build_run_paths(run_id)
    status = read_json(run_paths["status_path"], default={}) or {}
    status.update(changes)
    status["updated_at_utc"] = now_iso()
    write_json(run_paths["status_path"], status)
    with RUNS_LOCK:
        RUNS[str(run_id)] = status
    return status


def tail_log(log_path, max_lines=80):
    path = Path(log_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]


def launch_tracking_process(run_id, lineup_spec):
    run_paths = build_run_paths(run_id)
    run_paths["run_dir"].mkdir(parents=True, exist_ok=True)
    write_json(run_paths["spec_path"], lineup_spec)
    status = initial_status_payload(run_id, lineup_spec, run_paths)
    write_json(run_paths["status_path"], status)

    video_source = str(lineup_spec.get("video_source") or "").strip()
    if not video_source:
        raise LineupSpecError(
            "La interfaz necesita `video_source` para lanzar el tracking."
        )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "track.py"),
        video_source,
        "--lineup-spec",
        str(run_paths["spec_path"]),
    ]

    log_file = open(run_paths["log_path"], "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with RUNS_LOCK:
        RUNS[str(run_id)] = {
            **status,
            "status": "running",
            "pid": int(process.pid),
            "command": cmd,
        }
    update_status_file(
        run_id,
        status="running",
        pid=int(process.pid),
        command=cmd,
        started_at_utc=now_iso(),
    )

    def _watch():
        return_code = None
        error_message = None
        try:
            return_code = process.wait()
        except Exception as exc:
            error_message = str(exc)
        finally:
            log_file.close()

        if error_message is not None:
            update_status_file(
                run_id,
                status="failed",
                finished_at_utc=now_iso(),
                return_code=return_code,
                error=error_message,
            )
            return

        final_status = "completed" if int(return_code or 0) == 0 else "failed"
        update_status_file(
            run_id,
            status=final_status,
            finished_at_utc=now_iso(),
            return_code=int(return_code or 0),
        )

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    return read_json(run_paths["status_path"])


class InterfaceRequestHandler(BaseHTTPRequestHandler):
    server_version = "NarradorFutbolInterface/0.1"

    def log_message(self, format, *args):  # noqa: A003
        return

    def _send_json(self, payload, status=HTTPStatus.OK, include_body=True):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if include_body:
            self.wfile.write(data)

    def _send_file(self, file_path, include_body=True):
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            self._send_json(
                {"error": "Not found"},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if include_body:
            self.wfile.write(content)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _handle_get_run_status(self, run_id, include_body=True):
        run_paths = build_run_paths(run_id)
        status = read_json(run_paths["status_path"], default=None)
        if status is None:
            self._send_json(
                {"error": f"No existe la ejecución {run_id}."},
                status=HTTPStatus.NOT_FOUND,
                include_body=include_body,
            )
            return
        status["log_tail"] = tail_log(run_paths["log_path"])
        self._send_json(status, include_body=include_body)

    def _handle_request(self, include_body=True):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/formations":
            self._send_json(
                {"formations": get_formation_catalog()},
                include_body=include_body,
            )
            return

        if path == "/api/videos":
            self._send_json(
                {"videos": config_video_shortcuts()},
                include_body=include_body,
            )
            return

        if path == "/api/runs":
            run_items = []
            for run_dir in sorted(RUNS_ROOT.glob("*"), reverse=True):
                status = read_json(run_dir / "status.json", default=None)
                if status is not None:
                    run_items.append(status)
            self._send_json({"runs": run_items[:20]}, include_body=include_body)
            return

        if path.startswith("/api/runs/"):
            run_id = path.split("/api/runs/", 1)[1].strip("/")
            if not run_id:
                self._send_json(
                    {"error": "Run id inválido."},
                    status=HTTPStatus.BAD_REQUEST,
                    include_body=include_body,
                )
                return
            self._handle_get_run_status(run_id, include_body=include_body)
            return

        if path in {"/", ""}:
            self._send_file(STATIC_ROOT / "index.html", include_body=include_body)
            return

        static_candidate = (STATIC_ROOT / path.lstrip("/")).resolve()
        if STATIC_ROOT.resolve() not in static_candidate.parents and static_candidate != STATIC_ROOT.resolve():
            self._send_json(
                {"error": "Ruta no permitida."},
                status=HTTPStatus.FORBIDDEN,
                include_body=include_body,
            )
            return
        self._send_file(static_candidate, include_body=include_body)

    def do_GET(self):  # noqa: N802
        self._handle_request(include_body=True)

    def do_HEAD(self):  # noqa: N802
        self._handle_request(include_body=False)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/runs":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
            lineup_spec = validate_lineup_payload(payload)
            if not str(lineup_spec.get("video_source") or "").strip():
                raise LineupSpecError(
                    "Debes seleccionar un vídeo (`video_source`) antes de ejecutar."
                )
            run_id = uuid.uuid4().hex[:12]
            status = launch_tracking_process(run_id, lineup_spec)
            self._send_json(status, status=HTTPStatus.CREATED)
        except LineupSpecError as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
        except json.JSONDecodeError:
            self._send_json(
                {"error": "JSON inválido en la petición."},
                status=HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:  # pragma: no cover - defensivo
            self._send_json(
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interfaz web ligera para introducir alineaciones y lanzar tracking."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host a escuchar.")
    parser.add_argument("--port", type=int, default=8767, help="Puerto HTTP.")
    return parser.parse_args()


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), InterfaceRequestHandler)
    print(
        f"Interfaz disponible en http://{args.host}:{args.port} "
        f"(Python: {sys.executable})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
