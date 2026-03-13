import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Permite ejecutar `python scripts/track_partidos_posiciones.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import get_config

KAGGLE_DFL_DATASET = "saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv"


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


@dataclass(frozen=True)
class VideoTarget:
    label: str
    video_arg: str
    video_path: Path
    shortcut: Optional[str] = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta scripts/track.py en lote: vídeos de data/partidosPosiciones "
            "(config.yaml) o todos los .mp4 del dataset Kaggle DFL (train+test)."
        )
    )
    parser.add_argument(
        "--video-source",
        choices=("auto", "partidos-posiciones", "kaggle-all"),
        default="auto",
        help=(
            "Fuente de vídeos para el lote. "
            "'auto' prioriza Kaggle DFL si está disponible y, si no, usa "
            "data/partidosPosiciones."
        ),
    )
    parser.add_argument(
        "--kaggle-dataset",
        default=KAGGLE_DFL_DATASET,
        help=(
            "Identificador del dataset en Kaggle para --video-source kaggle-all "
            "(por defecto: saberghaderi/-dfl-bundesliga-460-mp4-videos-in-30sec-csv)."
        ),
    )
    parser.add_argument(
        "--kaggle-path",
        default=None,
        help=(
            "Ruta local al dataset Kaggle ya descargado. Si no se indica, "
            "se intenta descargar con kagglehub."
        ),
    )
    parser.add_argument(
        "--no-kaggle-download",
        action="store_true",
        help=(
            "No descargar desde Kaggle automáticamente. Útil junto a "
            "--kaggle-path."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Número de procesos en paralelo para lanzar track.py. "
            "Por seguridad en GPU se recomienda 1."
        ),
    )
    parser.add_argument(
        "--allow-gpu-parallel",
        action="store_true",
        help=(
            "Permite --workers > 1 aun detectando CUDA. Úsalo solo si "
            "tu VRAM aguanta múltiples procesos simultáneos."
        ),
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
        "--team-mode",
        choices=("reference", "auto-bootstrap"),
        default=None,
        help=(
            "Reenvía a track.py el modo de asignación de equipo por color. "
            "Si no se indica y la fuente es Kaggle, se usa auto-bootstrap."
        ),
    )
    parser.add_argument(
        "--team-bootstrap-frames",
        type=int,
        default=None,
        help=(
            "Reenvía a track.py los frames de bootstrap para el modo auto-bootstrap."
        ),
    )
    parser.add_argument(
        "--team-bootstrap-min-samples",
        type=int,
        default=None,
        help=(
            "Reenvía a track.py las muestras mínimas de color para cerrar bootstrap."
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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocesa también vídeos que ya tengan "
            "output/tracks_json/tracker/<video>_tracks.json."
        ),
    )
    parser.set_defaults(prompt_team_colors=False)
    return parser.parse_args()


def natural_video_key_sort(key):
    suffix = key.removeprefix("video_test_")
    if key.startswith("video_test_") and suffix.isdigit():
        return (0, int(suffix), key)
    return (1, key)


def natural_path_key(path_value):
    chunks = re.split(r"(\d+)", str(path_value).lower())
    return tuple(int(chunk) if chunk.isdigit() else chunk for chunk in chunks)


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


def collect_partidos_posiciones_targets(config):
    shortcuts = collect_partidos_posiciones_shortcuts(config)
    targets = []
    for shortcut in shortcuts:
        video_path = config.get_path("paths", "data", shortcut)
        targets.append(
            VideoTarget(
                label=shortcut,
                video_arg=shortcut,
                video_path=video_path,
                shortcut=shortcut,
            )
        )
    return targets


def resolve_kaggle_dataset_root(args):
    if args.kaggle_path:
        dataset_root = Path(args.kaggle_path).expanduser().resolve()
        if not dataset_root.exists():
            raise RuntimeError(
                f"La ruta de dataset indicada no existe: {dataset_root}"
            )
        return dataset_root

    if args.no_kaggle_download:
        raise RuntimeError(
            "No se indicó --kaggle-path y está activo --no-kaggle-download."
        )

    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar kagglehub. Instala dependencias con "
            "`pip install -r requirements.txt`."
        ) from exc

    dataset_root = Path(kagglehub.dataset_download(args.kaggle_dataset)).resolve()
    return dataset_root


def collect_kaggle_dataset_video_paths(dataset_root):
    all_mp4 = [path for path in dataset_root.rglob("*.mp4") if path.is_file()]
    split_mp4 = [
        path
        for path in all_mp4
        if any(part.lower() in {"train", "test"} for part in path.parts)
    ]
    selected = split_mp4 if split_mp4 else all_mp4
    selected.sort(key=lambda path: natural_path_key(path.relative_to(dataset_root)))
    return selected


def collect_kaggle_targets(args, strict=False):
    try:
        dataset_root = resolve_kaggle_dataset_root(args)
    except RuntimeError as exc:
        if strict:
            raise
        print(
            f"Aviso: no se pudo preparar dataset Kaggle ({exc}).",
            file=sys.stderr,
        )
        return [], None

    video_paths = collect_kaggle_dataset_video_paths(dataset_root)
    if not video_paths:
        message = f"No se encontraron .mp4 dentro de {dataset_root}"
        if strict:
            raise RuntimeError(message)
        print(f"Aviso: {message}.", file=sys.stderr)
        return [], dataset_root

    targets = []
    for video_path in video_paths:
        relative_path = video_path.relative_to(dataset_root).as_posix()
        targets.append(
            VideoTarget(
                label=f"kaggle:{relative_path}",
                video_arg=str(video_path),
                video_path=video_path,
            )
        )
    return targets, dataset_root


def sanitize_video_stem(raw_stem):
    stem = str(raw_stem).strip()
    if not stem:
        return "video"
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "video"


def expected_tracks_json_path(config, video_path):
    tracks_dir = config.get_path(
        "paths", "output", "tracks_json", create_if_missing=True
    ) / "tracker"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    sanitized_stem = sanitize_video_stem(video_path.stem)
    return tracks_dir / f"{sanitized_stem}_tracks.json"


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


def ask_team_colors_for_video(label, previous_value):
    print(f"\nColores para '{label}'")
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


def is_cuda_available():
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def main():
    args = parse_args()
    config = get_config()
    targets = []
    dataset_root = None
    effective_video_source = args.video_source

    try:
        if args.video_source == "partidos-posiciones":
            targets = collect_partidos_posiciones_targets(config)
            effective_video_source = "partidos-posiciones"
        elif args.video_source == "kaggle-all":
            targets, dataset_root = collect_kaggle_targets(args, strict=True)
            effective_video_source = "kaggle-all"
        else:
            strict_auto = bool(args.kaggle_path)
            targets, dataset_root = collect_kaggle_targets(args, strict=strict_auto)
            if not targets:
                targets = collect_partidos_posiciones_targets(config)
                effective_video_source = "partidos-posiciones"
            else:
                effective_video_source = "kaggle-all"
    except RuntimeError as exc:
        print(f"Error preparando lote: {exc}", file=sys.stderr)
        return 1

    if not targets:
        print("No se encontraron vídeos para procesar.", file=sys.stderr)
        return 1

    if dataset_root is not None and targets and targets[0].label.startswith("kaggle:"):
        print(f"Dataset Kaggle: {dataset_root}")
    print(f"Encontrados {len(targets)} vídeos para procesar.")

    workers = max(1, int(args.workers))
    if workers > 1 and is_cuda_available() and not args.allow_gpu_parallel:
        print(
            "CUDA detectada. Para evitar OOM/inestabilidad se usará workers=1. "
            "Si quieres forzar paralelo en GPU, añade --allow-gpu-parallel.",
            file=sys.stderr,
        )
        workers = 1

    if workers > 1 and not args.continue_on_error:
        print(
            "Aviso: en modo paralelo se ejecutan todos los vídeos lanzados "
            "aunque falle alguno (equivalente práctico a --continue-on-error).",
            file=sys.stderr,
        )

    success = []
    failed = []
    skipped = []
    last_team_colors = args.team_colors
    jobs = []
    total_targets = len(targets)

    if not args.team_colors and not args.no_default_color_plan:
        missing_shortcuts = [
            target.shortcut
            for target in targets
            if target.shortcut and target.shortcut not in DEFAULT_VIDEO_COLOR_PLAN
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

    for idx, target in enumerate(targets, start=1):
        tracks_json_path = expected_tracks_json_path(config, target.video_path)
        if tracks_json_path.exists() and not args.force:
            print(f"[{idx}/{total_targets}] {target.label}")
            print(f"  skip: ya existe {tracks_json_path}")
            skipped.append(target.label)
            continue

        current_team_colors = args.team_colors
        if (
            not current_team_colors
            and not args.no_default_color_plan
            and target.shortcut is not None
        ):
            current_team_colors = build_team_colors_override(target.shortcut)

        if args.prompt_team_colors and not args.dry_run:
            default_prompt_colors = current_team_colors or last_team_colors
            selected_colors = ask_team_colors_for_video(
                target.label, default_prompt_colors
            )
            if selected_colors == "quit":
                print("Lote finalizado por usuario.")
                break
            if selected_colors == "skip":
                print(f"Saltando '{target.label}'.")
                skipped.append(target.label)
                continue
            current_team_colors = selected_colors
            last_team_colors = selected_colors

        current_team_mode = args.team_mode
        if (
            current_team_mode is None
            and effective_video_source == "kaggle-all"
            and not current_team_colors
        ):
            current_team_mode = "auto-bootstrap"

        command = [sys.executable, "scripts/track.py", target.video_arg]
        if current_team_colors:
            command.extend(["--team-colors", current_team_colors])
        if current_team_mode:
            command.extend(["--team-mode", current_team_mode])
        if args.team_bootstrap_frames is not None:
            command.extend(
                ["--team-bootstrap-frames", str(int(args.team_bootstrap_frames))]
            )
        if args.team_bootstrap_min_samples is not None:
            command.extend(
                [
                    "--team-bootstrap-min-samples",
                    str(int(args.team_bootstrap_min_samples)),
                ]
            )

        cmd_text = " ".join(f'"{token}"' if " " in token else token for token in command)
        print(f"[{idx}/{total_targets}] {target.label}")
        if current_team_colors:
            print(f"  team-colors: {current_team_colors}")
        if current_team_mode:
            print(f"  team-mode: {current_team_mode}")
        print(f"  -> {cmd_text}")

        if args.dry_run:
            success.append(target.label)
            continue

        jobs.append((idx, target, command))

    if not args.dry_run:
        if workers <= 1:
            for idx, target, command in jobs:
                result = subprocess.run(command, cwd=config.project_root)
                if result.returncode == 0:
                    success.append(target.label)
                    continue

                failed.append(target.label)
                if not args.continue_on_error:
                    print(
                        f"Error procesando '{target.label}' (exit={result.returncode}). "
                        "Usa --continue-on-error para seguir con el resto.",
                        file=sys.stderr,
                    )
                    break
        else:
            future_to_job = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for idx, target, command in jobs:
                    future = executor.submit(
                        subprocess.run,
                        command,
                        cwd=config.project_root,
                    )
                    future_to_job[future] = (idx, target)

                for future in as_completed(future_to_job):
                    idx, target = future_to_job[future]
                    try:
                        result = future.result()
                        return_code = int(result.returncode)
                    except Exception as exc:  # pragma: no cover - protección runtime
                        return_code = 1
                        print(
                            f"[{idx}/{total_targets}] ERROR interno ejecutando "
                            f"'{target.label}': {exc}",
                            file=sys.stderr,
                        )
                    if return_code == 0:
                        success.append(target.label)
                        print(f"[{idx}/{total_targets}] OK {target.label}")
                    else:
                        failed.append(target.label)
                        print(
                            f"[{idx}/{total_targets}] FALLÓ {target.label} "
                            f"(exit={return_code})",
                            file=sys.stderr,
                        )

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
