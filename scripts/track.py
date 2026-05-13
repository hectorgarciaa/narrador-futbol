import argparse
import sys
from pathlib import Path

# Permite ejecutar `python scripts/track.py` sin instalar el paquete en editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.pipeline import run_tracking_pipeline


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
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Override explícito del modelo YOLO a usar en esta ejecución. "
            "Si no se indica, se usa paths.models.modelo_base de config.yaml."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directorio raíz donde guardar los artefactos de esta ejecución "
            "(tracks, summary, debug, vídeo, etc.)."
        ),
    )
    parser.add_argument(
        "--experiment-label",
        default=None,
        help=(
            "Etiqueta única para identificar la ejecución en métricas y manifiestos. "
            "Si no se indica, se usa el video_shortcut o la ruta del vídeo."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("runtime", "debug"),
        default=None,
        help=(
            "Sobrescribe tracking.execution_mode para esta ejecución. "
            "runtime desactiva trazas ricas; debug activa auditoría completa."
        ),
    )
    parser.add_argument(
        "--skip-render-video",
        action="store_true",
        help="No renderiza el vídeo anotado final; solo genera artefactos de tracking.",
    )
    parser.add_argument(
        "--skip-metrics-dataset",
        action="store_true",
        help=(
            "No actualiza data/posiciones_etiquetadas/common/tracking_metrics.csv. "
            "Util para benchmarks aislados."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximo numero de frames a procesar (util para pruebas rapidas).",
    )
    parser.add_argument(
        "--print-equipos",
        dest="print_equipos",
        action="store_true",
        default=True,
        help="Muestra nombres de equipos y distancias en el video anotado.",
    )
    parser.add_argument(
        "--no-print-equipos",
        dest="print_equipos",
        action="store_false",
        help="Oculta nombres de equipos y distancias en el video anotado.",
    )
    parser.add_argument(
        "--commentary",
        action="store_true",
        default=None,
        help="Activa generacion de comentarios (LLM + TTS) como fase del pipeline.",
    )
    parser.add_argument(
        "--no-commentary",
        dest="commentary",
        action="store_false",
        help="Desactiva generacion de comentarios.",
    )
    parser.add_argument(
        "--commentary-audio",
        action="store_true",
        default=None,
        help="Activa generacion de audio TTS en la fase de comentarios.",
    )
    parser.add_argument(
        "--no-commentary-audio",
        dest="commentary_audio",
        action="store_false",
        help="Desactiva generacion de audio TTS (solo texto).",
    )
    parser.add_argument(
        "--commentary-tts-backend",
        default=None,
        help="Backend TTS para comentarios (xtts, qwen, elevenlabs).",
    )
    parser.add_argument(
        "--commentary-llm-model",
        default=None,
        help="Modelo LLM para comentarios (default: gemma4:e2b).",
    )
    return parser.parse_args()


def main():
    return run_tracking_pipeline(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
