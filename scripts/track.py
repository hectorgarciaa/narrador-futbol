import argparse
import sys
from pathlib import Path

# Permite ejecutar `python scripts/track.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.tracking_cli import run_tracking_pipeline


def parse_args():
    """Parse CLI arguments for the tracking script."""
    parser = argparse.ArgumentParser(
        description=(
            "Run full tracking pipeline. Optionally pass a video shortcut from "
            "config.yaml paths.data (e.g. video_prueba_ajustado)."
        )
    )
    parser.add_argument(
        "video_shortcut",
        nargs="?",
        default=None,
        help=(
            "Optional key inside paths.data (config.yaml), e.g. video_prueba_ajustado. "
            "You can also pass a direct video path."
        ),
    )
    parser.add_argument(
        "--lineup-spec",
        default=None,
        help=(
            "Ruta a un JSON de alineaciones generado por la interfaz web. "
            "Permite definir equipos por color, formación y jugador por slot."
        ),
    )
    parser.add_argument(
        "--team-colors",
        default=None,
        help=(
            "Override de colores por terminal en formato "
            "'{Equipo:color, Otro:color}'. "
            "Acepta lenguaje natural, HEX (#RRGGBB) o RGB (255,255,255)."
        ),
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help=(
            "Mide y muestra tiempos por fase del pipeline en cada frame. "
            "No altera resultados, solo añade trazas de rendimiento."
        ),
    )
    return parser.parse_args()


def main():
    return run_tracking_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
