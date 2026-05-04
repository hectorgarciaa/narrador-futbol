from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .generator import (
    DEFAULT_COMMENTARY_TEMPERATURE,
    CommentaryEvent,
    CommentaryGenerator,
)
from .voice import (
    DEFAULT_ELEVENLABS_LANGUAGE_CODE,
    DEFAULT_ELEVENLABS_MODEL_ID,
    DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    DEFAULT_QWEN_CPP_THREADS,
    DEFAULT_QWEN_FEMALE_VOICE_DESIGN_PROMPT,
    DEFAULT_QWEN_REFERENCE_TEXT,
    DEFAULT_QWEN_VOICE_CLONE_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
    DEFAULT_QWEN_USE_FLASH_ATTENTION,
    DEFAULT_TTS_BACKEND,
    TTS_BACKEND_CHOICES,
    CommentaryAudioPipeline,
    build_voice_synthesizer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un comentario simulado de futbol usando llama.cpp con Gemma 4 GGUF."
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
        default="gemma4-q4ks-text",
        help="Alias del modelo servido por llama.cpp.",
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
        help="URL base del servidor llama.cpp. Si no se indica, usa LLAMA_CPP_BASE_URL o 127.0.0.1:8001.",
    )
    parser.add_argument(
        "--tts-backend",
        choices=TTS_BACKEND_CHOICES,
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
        "--female-speaker-wav",
        action="append",
        default=None,
        help=(
            "Ruta a un WAV femenino para alternar voces con XTTS. "
            "Se puede repetir."
        ),
    )
    parser.add_argument(
        "--alternate-voices",
        action="store_true",
        help=(
            "Usa una voz masculina y una femenina con aleatoriedad controlada "
            "y maximo tres comentarios seguidos de la misma voz. Con Qwen disena "
            "ambas; con XTTS usa --female-speaker-wav o una referencia femenina "
            "de Qwen cacheada."
        ),
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
        "--qwen-female-voice-design-prompt",
        default=DEFAULT_QWEN_FEMALE_VOICE_DESIGN_PROMPT,
        help="Prompt de diseno de voz femenina para alternar comentaristas.",
    )
    parser.add_argument(
        "--qwen-reference-text",
        default=DEFAULT_QWEN_REFERENCE_TEXT,
        help="Texto que se usa para crear el clip de referencia con VoiceDesign.",
    )
    parser.add_argument(
        "--generate-female-qwen-reference",
        action="store_true",
        help=(
            "Si usas XTTS con --alternate-voices y no pasas --female-speaker-wav, "
            "genera/cachea primero la referencia femenina con Qwen VoiceDesign."
        ),
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
        "--elevenlabs-api-key",
        default=None,
        help="API key de ElevenLabs. Si no se indica, usa ELEVENLABS_API_KEY.",
    )
    parser.add_argument(
        "--elevenlabs-voice-id",
        default=None,
        help="ID de la voz de ElevenLabs. Si no se indica, usa ELEVENLABS_VOICE_ID.",
    )
    parser.add_argument(
        "--elevenlabs-female-voice-id",
        default=None,
        help=(
            "ID de voz femenina de ElevenLabs para --alternate-voices. "
            "Si no se indica, usa ELEVENLABS_FEMALE_VOICE_ID."
        ),
    )
    parser.add_argument(
        "--elevenlabs-model-id",
        default=None,
        help=(
            "Modelo de ElevenLabs para TTS streaming. "
            f"Default/env: ELEVENLABS_MODEL_ID o {DEFAULT_ELEVENLABS_MODEL_ID}."
        ),
    )
    parser.add_argument(
        "--elevenlabs-output-format",
        default=None,
        help=(
            "Formato de salida para ElevenLabs streaming. "
            f"Default/env: ELEVENLABS_OUTPUT_FORMAT o {DEFAULT_ELEVENLABS_OUTPUT_FORMAT}."
        ),
    )
    parser.add_argument(
        "--elevenlabs-language-code",
        default=None,
        help=(
            "Codigo de idioma enviado a ElevenLabs. "
            f"Default/env: ELEVENLABS_LANGUAGE_CODE o {DEFAULT_ELEVENLABS_LANGUAGE_CODE}."
        ),
    )
    parser.add_argument(
        "--elevenlabs-stability",
        type=float,
        default=None,
        help="Voice setting opcional `stability` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-similarity-boost",
        type=float,
        default=None,
        help="Voice setting opcional `similarity_boost` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-style",
        type=float,
        default=None,
        help="Voice setting opcional `style` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-speed",
        type=float,
        default=None,
        help="Voice setting opcional `speed` de ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-use-speaker-boost",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activa o desactiva `use_speaker_boost` en ElevenLabs.",
    )
    parser.add_argument(
        "--elevenlabs-optimize-streaming-latency",
        type=int,
        choices=range(0, 5),
        default=None,
        metavar="{0,1,2,3,4}",
        help="Optimizacion de latencia streaming de ElevenLabs.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Genera solo el comentario, sin convertirlo a audio.",
    )
    parser.add_argument(
        "--prepare-voice-only",
        action="store_true",
        help=(
            "Prepara/cachea las voces configuradas y termina sin generar un comentario."
        ),
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
    CommentaryGenerator,
    Any | None,
    CommentaryAudioPipeline | None,
]:
    generator = CommentaryGenerator(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    if args.text_only:
        return generator, None, None

    voice_synthesizer = build_voice_synthesizer(
        tts_backend=args.tts_backend,
        speaker_wavs=args.speaker_wav,
        female_speaker_wavs=args.female_speaker_wav,
        alternate_voices=args.alternate_voices,
        tts_model=args.tts_model,
        tts_language=args.tts_language,
        use_gpu=False if args.cpu else None,
        split_sentences=not args.no_split_sentences,
        qwen_design_model_name=args.qwen_design_model,
        qwen_clone_model_name=args.qwen_clone_model,
        qwen_voice_design_prompt=args.qwen_voice_design_prompt,
        qwen_female_voice_design_prompt=args.qwen_female_voice_design_prompt,
        qwen_reference_text=args.qwen_reference_text,
        generate_female_qwen_reference=args.generate_female_qwen_reference,
        qwen_use_flash_attention=(
            bool(args.qwen_flash_attn) and not bool(args.qwen_no_flash_attn)
        ),
        qwen_cpp_threads=args.qwen_cpp_threads,
        qwen_cpp_repo_dir=args.qwen_cpp_repo_dir,
        qwen_cpp_model_dir=args.qwen_cpp_model_dir,
        elevenlabs_api_key=args.elevenlabs_api_key,
        elevenlabs_voice_id=args.elevenlabs_voice_id,
        elevenlabs_female_voice_id=args.elevenlabs_female_voice_id,
        elevenlabs_model_id=args.elevenlabs_model_id,
        elevenlabs_output_format=args.elevenlabs_output_format,
        elevenlabs_language_code=args.elevenlabs_language_code,
        elevenlabs_stability=args.elevenlabs_stability,
        elevenlabs_similarity_boost=args.elevenlabs_similarity_boost,
        elevenlabs_style=args.elevenlabs_style,
        elevenlabs_speed=args.elevenlabs_speed,
        elevenlabs_use_speaker_boost=args.elevenlabs_use_speaker_boost,
        elevenlabs_optimize_streaming_latency=(
            args.elevenlabs_optimize_streaming_latency
        ),
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

    if args.prepare_voice_only:
        if voice_synthesizer is None:
            raise RuntimeError("--prepare-voice-only requiere un backend de voz.")
        voice_synthesizer.prepare(warmup_text=args.qwen_reference_text)
        print(f"TTS_MODEL={getattr(voice_synthesizer, 'model_name', 'unknown')}")
        speaker_wavs = getattr(voice_synthesizer, "speaker_wavs", ()) or ()
        for index, speaker_wav in enumerate(speaker_wavs, start=1):
            print(f"SPEAKER_WAV_{index}={speaker_wav}")
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
                        "voice_label": result.voice_label,
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
    if result.voice_label:
        print(f"VOICE_LABEL={result.voice_label}")
    if args.print_timings:
        print(f"LLM_SEC={result.llm_seconds or 0.0:.3f}")
        print(f"TTS_SEC={result.tts_seconds or 0.0:.3f}")
        print(f"TOTAL_SEC={result.total_seconds or 0.0:.3f}")


if __name__ == "__main__":
    main()
