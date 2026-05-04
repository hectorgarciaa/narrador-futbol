from __future__ import annotations

import argparse
import json

from .generator import (
    DEFAULT_COMMENTARY_TEMPERATURE,
    CommentaryEvent,
    CommentaryGenerator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalua el backend LLM de comentarios con llama.cpp sin pasar por TTS."
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
        help="Temperatura de muestreo.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p de muestreo.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="URL base del servidor llama.cpp (por defecto LLAMA_CPP_BASE_URL o 127.0.0.1:8001).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximo de tokens nuevos.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Numero de generaciones consecutivas para comparar salidas.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Imprime el system prompt y el user prompt usados.",
    )
    parser.add_argument(
        "--show-payload",
        action="store_true",
        help="Imprime el payload o parametros que se envian al backend.",
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Imprime la respuesta cruda devuelta por el backend.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Sobrescribe el system prompt generado por defecto.",
    )
    parser.add_argument(
        "--user-prompt",
        default=None,
        help="Sobrescribe el user prompt generado por defecto.",
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
        field_zone="la frontal del area",
        action_index=30,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        raise ValueError("`--repeat` debe ser >= 1.")

    event = load_event(args)
    generator = CommentaryGenerator(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        base_url=args.base_url,
        max_tokens=args.max_new_tokens,
    )

    event, default_system_prompt, default_user_prompt = generator.build_prompts(event)
    system_prompt = args.system_prompt or default_system_prompt
    user_prompt = args.user_prompt or default_user_prompt

    if args.show_prompts:
        print("=== SYSTEM PROMPT ===")
        print(system_prompt)
        print()
        print("=== USER PROMPT ===")
        print(user_prompt)
        print()

    for idx in range(args.repeat):
        result = generator.run_llm(
            event,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if args.repeat > 1:
            print(f"=== RUN {idx + 1} ===")
        print(f"RAW_COMMENTARY={result.raw_commentary}")
        print(f"CLEANED_COMMENTARY={result.cleaned_commentary}")
        print(f"FINAL_COMMENTARY={result.final_commentary}")
        if result.total_duration_seconds is not None:
            print(f"TOTAL_DURATION_SEC={result.total_duration_seconds:.3f}")
        if args.show_payload:
            print("REQUEST_PAYLOAD=")
            print(
                json.dumps(
                    result.request_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        if args.show_raw_response:
            print("RAW_RESPONSE=")
            print(
                json.dumps(
                    result.raw_response,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        if idx + 1 < args.repeat:
            print()
        print("=== USER PROMPT ===")
        print(user_prompt)
        print()

    for idx in range(args.repeat):
        result = generator.run_llm(
            event,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if args.repeat > 1:
            print(f"=== RUN {idx + 1} ===")
        print(f"RAW_COMMENTARY={result.raw_commentary}")
        print(f"CLEANED_COMMENTARY={result.cleaned_commentary}")
        print(f"FINAL_COMMENTARY={result.final_commentary}")
        if result.total_duration_seconds is not None:
            print(f"TOTAL_DURATION_SEC={result.total_duration_seconds:.3f}")
        if args.show_payload:
            print("REQUEST_PAYLOAD=")
            print(
                json.dumps(
                    result.request_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        if args.show_raw_response:
            print("RAW_RESPONSE=")
            print(
                json.dumps(
                    result.raw_response,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        if idx + 1 < args.repeat:
            print()


if __name__ == "__main__":
    main()
