import argparse
import subprocess
import sys
from pathlib import Path

# Permite ejecutar `python scripts/track_partidos_posiciones.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import get_config


DEFAULT_VIDEO_COLOR_PLAN = {
    "video_test_1": ("rojo", "negro"),
    "video_test_2": ("blanco", "amarillo"),
    "video_test_11": ("gris", "negro"),
    "video_test_12": ("rojo", "negro"),
    "video_test_13": ("blanco", "azul"),
    "video_test_14": ("blanco", "amarillo"),
    "video_test_15": ("blanco", "rojo"),
    "video_test_16": ("blanco", "negro"),
    "video_test_18": ("amarillo", "gris"),
    "video_test_20": ("amarillo", "gris"),
    "video_test_22": ("rojo", "negro"),
    "video_test_26": ("blanco", "negro"),
    "video_test_27": ("amarillo", "gris"),
    # Corregido desde "ojo y negro" (typo) a "rojo y negro".
    "video_test_29": ("rojo", "negro"),
}

DEFAULT_TEAM_NAME_BY_COLOR = {
    "blanco": "Real Madrid",
    "negro": "Equipo Negro",
    "rojo": "Equipo Rojo",
    "amarillo": "Equipo Amarillo",
    "gris": "Equipo Gris",
    "azul": "Equipo Azul",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta scripts/track.py para todos los vídeos definidos en "
            "data/partidosPosiciones dentro de config.yaml."
        )
    )
    parser.add_argument(
        "--team-colors",
        default=None,
        help=(
            "Se reenvía tal cual a track.py --team-colors. "
            "Ejemplo: \"{Madrid:blanco, Wolsfburgo:verde-claro}\"."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Si un vídeo falla, continúa con el siguiente.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra los comandos sin ejecutarlos.",
    )
    parser.add_argument(
        "--prompt-team-colors",
        dest="prompt_team_colors",
        action="store_true",
        help=(
            "Pide por terminal colores de equipo para cada vídeo. "
            "Desactivado por defecto."
        ),
    )
    parser.add_argument(
        "--no-prompt-team-colors",
        dest="prompt_team_colors",
        action="store_false",
        help="Desactiva la petición interactiva de colores por vídeo.",
    )
    parser.add_argument(
        "--no-default-color-plan",
        action="store_true",
        help=(
            "Desactiva el plan fijo de colores por vídeo para data/partidosPosiciones. "
            "Si lo usas, debes pasar --team-colors o activar --prompt-team-colors."
        ),
    )
    parser.set_defaults(prompt_team_colors=False)
    return parser.parse_args()


def natural_video_key_sort(key):
    suffix = key.removeprefix("video_test_")
    if key.startswith("video_test_") and suffix.isdigit():
        return (0, int(suffix), key)
    return (1, key)


def collect_partidos_posiciones_shortcuts(config):
    data_paths = config.paths.get("data", {})
    partidos_dir = (config.project_root / "data" / "partidosPosiciones").resolve()
    selected = []
    for key, raw_path in data_paths.items():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (config.project_root / candidate).resolve()
        if partidos_dir == candidate.parent or partidos_dir in candidate.parents:
            selected.append(key)
    return sorted(selected, key=natural_video_key_sort)


def normalize_color_name(color_name):
    normalized = str(color_name).strip().lower()
    normalized = normalized.replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized


def build_team_colors_override(shortcut):
    colors = DEFAULT_VIDEO_COLOR_PLAN.get(shortcut)
    if not colors:
        return None

    entries = []
    used_team_names = set()
    for idx, color in enumerate(colors, start=1):
        normalized_color = normalize_color_name(color)
        team_name = DEFAULT_TEAM_NAME_BY_COLOR.get(
            normalized_color,
            f"Equipo {normalized_color.replace('-', ' ').title()}",
        )
        if team_name in used_team_names:
            team_name = f"{team_name} {idx}"
        used_team_names.add(team_name)
        entries.append(f"{team_name}:{color}")
    return "{" + ", ".join(entries) + "}"


def ask_team_colors_for_video(shortcut, previous_value):
    print(f"\nColores para '{shortcut}'")
    print("Formato esperado: {Equipo:color, Otro:color}")
    print("Comandos: 'skip' para saltar, 'quit' para terminar lote.")
    if previous_value:
        print(f"Enter vacío = reutilizar últimos colores: {previous_value}")
    else:
        print("Enter vacío no permitido (no hay colores previos).")

    while True:
        raw_value = input("> ").strip()
        normalized = raw_value.lower()
        if normalized in {"quit", "exit", "q"}:
            return "quit"
        if normalized in {"skip", "s"}:
            return "skip"
        if raw_value:
            return raw_value
        if previous_value:
            return previous_value
        print("Introduce colores o usa 'skip'/'quit'.")


def main():
    args = parse_args()
    config = get_config()
    shortcuts = collect_partidos_posiciones_shortcuts(config)
    if not shortcuts:
        print(
            "No se encontraron shortcuts de data/partidosPosiciones en config.yaml.",
            file=sys.stderr,
        )
        return 1

    print(f"Encontrados {len(shortcuts)} shortcuts: {', '.join(shortcuts)}")
    success = []
    failed = []
    skipped = []
    last_team_colors = args.team_colors

    if not args.team_colors and not args.no_default_color_plan:
        missing_shortcuts = [
            shortcut for shortcut in shortcuts if shortcut not in DEFAULT_VIDEO_COLOR_PLAN
        ]
        if missing_shortcuts:
            print(
                "Falta plan de colores para: "
                + ", ".join(missing_shortcuts)
                + ". Añádelos a DEFAULT_VIDEO_COLOR_PLAN o ejecuta con "
                "--team-colors / --prompt-team-colors.",
                file=sys.stderr,
            )
            return 1

    for idx, shortcut in enumerate(shortcuts, start=1):
        current_team_colors = args.team_colors
        if not current_team_colors and not args.no_default_color_plan:
            current_team_colors = build_team_colors_override(shortcut)

        if args.prompt_team_colors and not args.dry_run:
            default_prompt_colors = current_team_colors or last_team_colors
            selected_colors = ask_team_colors_for_video(shortcut, default_prompt_colors)
            if selected_colors == "quit":
                print("Lote finalizado por usuario.")
                break
            if selected_colors == "skip":
                print(f"Saltando '{shortcut}'.")
                skipped.append(shortcut)
                continue
            current_team_colors = selected_colors
            last_team_colors = selected_colors

        command = [sys.executable, "scripts/track.py", shortcut]
        if current_team_colors:
            command.extend(["--team-colors", current_team_colors])

        cmd_text = " ".join(f'"{token}"' if " " in token else token for token in command)
        print(f"[{idx}/{len(shortcuts)}] {shortcut}")
        if current_team_colors:
            print(f"  team-colors: {current_team_colors}")
        print(f"  -> {cmd_text}")

        if args.dry_run:
            success.append(shortcut)
            continue

        result = subprocess.run(command, cwd=config.project_root)
        if result.returncode == 0:
            success.append(shortcut)
            continue

        failed.append(shortcut)
        if not args.continue_on_error:
            print(
                f"Error procesando '{shortcut}' (exit={result.returncode}). "
                "Usa --continue-on-error para seguir con el resto.",
                file=sys.stderr,
            )
            break

    print("\nResumen:")
    print(f"- OK: {len(success)}")
    print(f"- Fallidos: {len(failed)}")
    print(f"- Saltados: {len(skipped)}")
    if skipped:
        print(f"- Lista saltados: {', '.join(skipped)}")
    if failed:
        print(f"- Lista fallidos: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
