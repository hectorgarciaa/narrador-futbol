from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2


@dataclass(slots=True)
class DeferredCommentaryAssemblyResult:
    source_video_path: Path
    output_video_path: Path
    commentary_track_path: Path
    event_count: int
    video_duration_seconds: float


def _load_ffmpeg_executable() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError as exc:
        raise RuntimeError(
            "Falta `imageio-ffmpeg` para ensamblar el audio diferido dentro del MP4 final."
        ) from exc
    return str(get_ffmpeg_exe())


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    entries = []
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        return entries
    with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _event_audio_entries(manifest_path: str | Path) -> list[dict[str, Any]]:
    entries = []
    for item in _read_jsonl(manifest_path):
        audio_path = item.get("audio_path")
        event = item.get("event") or {}
        if not audio_path:
            continue
        resolved_audio_path = Path(audio_path).expanduser().resolve()
        if not resolved_audio_path.exists():
            continue
        try:
            event_time_s = float(event.get("event_time_s") or 0.0)
        except (TypeError, ValueError):
            event_time_s = 0.0
        entries.append(
            {
                "event_time_s": max(0.0, event_time_s),
                "audio_path": resolved_audio_path,
            }
        )
    entries.sort(key=lambda item: (item["event_time_s"], str(item["audio_path"])))
    return entries


def probe_video_duration_seconds(video_path: str | Path) -> float:
    resolved_video_path = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(resolved_video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo: {resolved_video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0.0 or frame_count <= 0.0:
            raise RuntimeError(
                f"No se pudo estimar la duración del vídeo: {resolved_video_path}"
            )
        return frame_count / fps
    finally:
        cap.release()


def build_commentary_track_from_manifest(
    manifest_path: str | Path,
    output_audio_path: str | Path,
    *,
    video_duration_seconds: float,
    sample_rate: int = 24000,
    channel_layout: str = "mono",
) -> tuple[Path, int]:
    audio_entries = _event_audio_entries(manifest_path)
    if not audio_entries:
        raise RuntimeError("El manifiesto no contiene audios para construir la pista.")

    ffmpeg_exe = _load_ffmpeg_executable()
    output_path = Path(output_audio_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-t",
        f"{float(video_duration_seconds):.3f}",
        "-i",
        f"anullsrc=r={int(sample_rate)}:cl={channel_layout}",
    ]
    for item in audio_entries:
        command.extend(["-i", str(item["audio_path"])])

    filter_parts = []
    mix_inputs = ["[0:a]"]
    for input_index, item in enumerate(audio_entries, start=1):
        delay_ms = max(0, int(round(float(item["event_time_s"]) * 1000.0)))
        label = f"a{input_index}"
        filter_parts.append(
            f"[{input_index}:a]aformat=sample_rates={int(sample_rate)}:"
            f"channel_layouts={channel_layout},adelay={delay_ms}|{delay_ms}[{label}]"
        )
        mix_inputs.append(f"[{label}]")

    filter_parts.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0[outa]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outa]",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    if not output_path.exists():
        raise RuntimeError(
            f"No se pudo generar la pista de comentarios esperada: {output_path}"
        )
    return output_path, len(audio_entries)


def mux_commentary_track_into_video(
    video_path: str | Path,
    commentary_track_path: str | Path,
    output_video_path: str | Path,
) -> Path:
    ffmpeg_exe = _load_ffmpeg_executable()
    source_video_path = Path(video_path).expanduser().resolve()
    track_path = Path(commentary_track_path).expanduser().resolve()
    final_output_path = Path(output_video_path).expanduser().resolve()
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    if final_output_path == source_video_path:
        temp_output_path = final_output_path.with_name(
            final_output_path.stem + "_commentary_tmp.mp4"
        )
    else:
        temp_output_path = final_output_path

    command = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_video_path),
        "-i",
        str(track_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(temp_output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    if final_output_path == source_video_path:
        temp_output_path.replace(final_output_path)
    if not final_output_path.exists():
        raise RuntimeError(
            f"No se pudo generar el vídeo final con comentarios: {final_output_path}"
        )
    return final_output_path


def assemble_deferred_commentary_video(
    manifest_path: str | Path,
    source_video_path: str | Path,
    commentary_track_path: str | Path,
    output_video_path: str | Path,
) -> DeferredCommentaryAssemblyResult:
    resolved_video_path = Path(source_video_path).expanduser().resolve()
    if not resolved_video_path.exists():
        raise FileNotFoundError(
            f"No existe el vídeo procesado para mezclar comentarios: {resolved_video_path}"
        )

    video_duration_seconds = probe_video_duration_seconds(resolved_video_path)
    built_track_path, event_count = build_commentary_track_from_manifest(
        manifest_path,
        commentary_track_path,
        video_duration_seconds=video_duration_seconds,
    )
    final_video_path = mux_commentary_track_into_video(
        resolved_video_path,
        built_track_path,
        output_video_path,
    )
    return DeferredCommentaryAssemblyResult(
        source_video_path=resolved_video_path,
        output_video_path=final_video_path,
        commentary_track_path=built_track_path,
        event_count=event_count,
        video_duration_seconds=video_duration_seconds,
    )
