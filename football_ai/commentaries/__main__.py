from __future__ import annotations

import argparse
import json

from .generator import CommentaryEvent, OllamaCommentaryGenerator
from .voice import CommentaryAudioPipeline, XTTSVoiceSynthesizer


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
        default="tinyllama:1.1b",
        help="Modelo de Ollama a usar.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.95,
        help="Temperatura alta para variar el estilo del comentarista.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="URL base del servidor de Ollama. Si no se indica, usa OLLAMA_HOST o 127.0.0.1:11434.",
    )
    parser.add_argument(
        "--speaker-wav",
        action="append",
        default=None,
        help="Ruta a un WAV de referencia para clonar la voz. Se puede repetir.",
    )
    parser.add_argument(
        "--audio-out",
        default=None,
        help="Ruta del WAV de salida. Si no se indica, se genera en output/commentaries/audio/.",
    )
    parser.add_argument(
        "--tts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Modelo de Coqui TTS a usar para clonar la voz.",
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
        "--text-only",
        action="store_true",
        help="Genera solo el comentario, sin convertirlo a audio.",
    )
    parser.add_argument(
        "--no-split-sentences",
        action="store_true",
        help="Desactiva la division en frases antes de sintetizar.",
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
        field_zone="frontal del area",
        action_index=30,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    event = load_event(args)
    generator = OllamaCommentaryGenerator(
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
    )
    if args.text_only:
        result = generator.generate(event)
        print(result.commentary)
        return

    voice_synthesizer = XTTSVoiceSynthesizer(
        model_name=args.tts_model,
        speaker_wavs=args.speaker_wav,
        language=args.tts_language,
        use_gpu=False if args.cpu else None,
        split_sentences=not args.no_split_sentences,
    )
    pipeline = CommentaryAudioPipeline(
        commentary_generator=generator,
        voice_synthesizer=voice_synthesizer,
    )
    result = pipeline.generate_to_file(event, audio_path=args.audio_out)
    print(result.commentary)
    print(f"AUDIO_FILE={result.audio_path}")


if __name__ == "__main__":
    main()
