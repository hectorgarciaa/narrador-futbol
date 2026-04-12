from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .generator import (
    DEFAULT_COMMENTARY_TEMPERATURE,
    CommentaryEvent,
    OllamaCommentaryGenerator,
)
from .server import DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT, create_http_server
from .voice import (
    DEFAULT_QWEN_CPP_THREADS,
    DEFAULT_QWEN_REFERENCE_TEXT,
    DEFAULT_QWEN_VOICE_CLONE_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
    DEFAULT_QWEN_USE_FLASH_ATTENTION,
    DEFAULT_TTS_BACKEND,
    CommentaryAudioPipeline,
    build_voice_synthesizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un comentario simulado de futbol usando Ollama."
    )
    parser.add_argument(
        "--event-json",
        default=None,
        help="Evento JSON inline con los campos del comentario.",
    )
    parser.add_argument(
        "--event-file",
        default=None,
        help="Ruta a un JSON con el evento.",
    )
    parser.add_argument(
        "--model",
        default="gemma4:e2b",
        help="Modelo de Ollama a usar.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_COMMENTARY_TEMPERATURE,
        help="Temperatura de muestreo del modelo.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="URL base del servidor de Ollama. Si no se indica, usa OLLAMA_HOST o 127.0.0.1:11434.",
    )
    parser.add_argument(
        "--tts-backend",
        choices=["xtts", "qwen", "qwen_cpp"],
        default=DEFAULT_TTS_BACKEND,
        help="Backend de voz a usar.",
    )
    parser.add_argument(
        "--speaker-wav",
        action="append",
        default=None,
        help="Ruta a un WAV de referencia para clonar la voz con XTTS. Se puede repetir.",
    )
    parser.add_argument(
        "--audio-out",
        default=None,
        help="Ruta del WAV de salida. Si no se indica, se genera en output/commentaries/audio/.",
    )
    parser.add_argument(
        "--tts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Modelo de Coqui TTS a usar cuando `--tts-backend xtts`.",
    )
    parser.add_argument(
        "--tts-language",
        default="es",
        help="Idioma de la sintesis.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Fuerza la sintesis de voz en CPU aunque haya CUDA.",
    )
    parser.add_argument(
        "--qwen-design-model",
        default=DEFAULT_QWEN_VOICE_DESIGN_MODEL,
        help="Modelo de Qwen3-TTS VoiceDesign.",
    )
    parser.add_argument(
        "--qwen-clone-model",
        default=DEFAULT_QWEN_VOICE_CLONE_MODEL,
        help="Modelo de Qwen3-TTS Base para reutilizar la voz disenada.",
    )
    parser.add_argument(
        "--qwen-voice-design-prompt",
        default=DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
        help="Prompt de diseno de voz para Qwen3-TTS VoiceDesign.",
    )
    parser.add_argument(
        "--qwen-reference-text",
        default=DEFAULT_QWEN_REFERENCE_TEXT,
        help="Texto que se usa para crear el clip de referencia con VoiceDesign.",
    )
    parser.add_argument(
        "--qwen-flash-attn",
        action="store_true",
        default=DEFAULT_QWEN_USE_FLASH_ATTENTION,
        help="Activa FlashAttention al cargar Qwen3-TTS.",
    )
    parser.add_argument(
        "--qwen-no-flash-attn",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--qwen-cpp-threads",
        type=int,
        default=DEFAULT_QWEN_CPP_THREADS,
        help="Numero de hilos para qwen3-tts.cpp.",
    )
    parser.add_argument(
        "--qwen-cpp-repo-dir",
        default=None,
        help="Ruta al clone local de qwen3-tts.cpp.",
    )
    parser.add_argument(
        "--qwen-cpp-model-dir",
        default=None,
        help="Ruta al directorio con los modelos GGUF de qwen3-tts.cpp.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Genera solo el comentario, sin convertirlo a audio.",
    )
    parser.add_argument(
        "--no-split-sentences",
        action="store_true",
        help="Desactiva la division en frases antes de sintetizar.",
    )
    parser.add_argument(
        "--print-timings",
        action="store_true",
        help="Imprime tiempos de LLM, TTS y total para la ejecucion actual.",
    )
    parser.add_argument(
        "--jsonl-stdin",
        action="store_true",
        help=(
            "Mantiene el proceso caliente y lee eventos JSON, uno por linea, "
            "desde stdin. Devuelve una linea JSON por evento."
        ),
    )
    parser.add_argument(
        "--http-server",
        action="store_true",
        help="Arranca un servidor HTTP local para recibir eventos via POST.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        help="Host para el servidor HTTP local.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Puerto para el servidor HTTP local.",
    )
    return parser


def load_event(args: argparse.Namespace) -> CommentaryEvent:
    if args.event_json:
        return CommentaryEvent.from_dict(json.loads(args.event_json))
    if args.event_file:
        with open(args.event_file, "r", encoding="utf-8") as f:
            return CommentaryEvent.from_dict(json.load(f))
    return CommentaryEvent(
        action="gol",
        player_name="Modric",
        player_position="MC",
        event_time_s=187.0,
        team_name="Real Madrid",
        opponent_team_name="Wolfsburgo",
        field_zone="frontal del area",
        action_index=30,
    )


def build_runtime(
    args: argparse.Namespace,
) -> tuple[
    OllamaCommentaryGenerator,
    Any | None,
    CommentaryAudioPipeline | None,
]:
    generator = OllamaCommentaryGenerator(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    if args.text_only:
        return generator, None, None

    voice_synthesizer = build_voice_synthesizer(
        tts_backend=args.tts_backend,
        speaker_wavs=args.speaker_wav,
        tts_model=args.tts_model,
        tts_language=args.tts_language,
        use_gpu=False if args.cpu else None,
        split_sentences=not args.no_split_sentences,
        qwen_design_model_name=args.qwen_design_model,
        qwen_clone_model_name=args.qwen_clone_model,
        qwen_voice_design_prompt=args.qwen_voice_design_prompt,
        qwen_reference_text=args.qwen_reference_text,
        qwen_use_flash_attention=(
            bool(args.qwen_flash_attn) and not bool(args.qwen_no_flash_attn)
        ),
        qwen_cpp_threads=args.qwen_cpp_threads,
        qwen_cpp_repo_dir=args.qwen_cpp_repo_dir,
        qwen_cpp_model_dir=args.qwen_cpp_model_dir,
    )
    pipeline = CommentaryAudioPipeline(
        commentary_generator=generator,
        voice_synthesizer=voice_synthesizer,
    )
    return generator, voice_synthesizer, pipeline


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generator, voice_synthesizer, pipeline = build_runtime(args)

    if args.http_server:
        if pipeline is not None:
            pipeline.prepare()
        else:
            generator.prepare()
        server = create_http_server(
            commentary_generator=generator,
            audio_pipeline=pipeline,
            host=args.host,
            port=args.port,
            text_only=args.text_only,
        )
        print(
            f"Commentaries HTTP server listening on http://{args.host}:{args.port}",
            flush=True,
        )
        server.serve_forever()
        return

    if args.jsonl_stdin:
        if pipeline is not None:
            pipeline.prepare()
        else:
            generator.prepare()
        print("READY", file=sys.stderr, flush=True)
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = CommentaryEvent.from_dict(json.loads(line))
                if args.text_only:
                    started = time.perf_counter()
                    result = generator.generate(event)
                    payload = {
                        "commentary": result.commentary,
                        "model": result.model,
                        "llm_seconds": round(time.perf_counter() - started, 3),
                    }
                else:
                    result = pipeline.generate_to_file(event)
                    payload = {
                        "commentary": result.commentary,
                        "audio_path": str(result.audio_path),
                        "model": result.commentary_result.model,
                        "llm_seconds": round(result.llm_seconds or 0.0, 3),
                        "tts_seconds": round(result.tts_seconds or 0.0, 3),
                        "total_seconds": round(result.total_seconds or 0.0, 3),
                    }
            except Exception as exc:
                payload = {"error": str(exc)}
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        return

    event = load_event(args)
    if args.text_only:
        started = time.perf_counter()
        result = generator.generate(event)
        print(result.commentary)
        if args.print_timings:
            print(f"LLM_SEC={time.perf_counter() - started:.3f}")
        return

    if pipeline is None:
        raise RuntimeError("No se pudo construir el pipeline de audio.")
    result = pipeline.generate_to_file(event, audio_path=args.audio_out)
    print(result.commentary)
    print(f"AUDIO_FILE={result.audio_path}")
    if args.print_timings:
        print(f"LLM_SEC={result.llm_seconds or 0.0:.3f}")
        print(f"TTS_SEC={result.tts_seconds or 0.0:.3f}")
        print(f"TOTAL_SEC={result.total_seconds or 0.0:.3f}")


if __name__ == "__main__":
    main()
