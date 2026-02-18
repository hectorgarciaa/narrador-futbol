# football_ai

Paquete Python principal del sistema de narración de fútbol con IA. Contiene toda la lógica de negocio organizada en submódulos independientes y reutilizables. Los scripts en `scripts/` y los notebooks en `experiments/` actúan como clientes de este paquete.

## Estructura

```
football_ai/
├── __init__.py
├── core/           # Configuración global, logging centralizado, serialización
├── detection/      # Inferencia YOLO (jugadores, árbitros, balón)
├── tracking/       # Pipeline completo de tracking multi-objeto con ByteTrack
├── identification/ # Identificación de equipo por color de camiseta (KMeans)
├── evaluation/     # Cálculo de métricas por track y comparación de experimentos
└── visualization/  # Generación de video anotado con bounding boxes
```

## Responsabilidades de cada módulo

| Módulo | Responsabilidad principal | Clases clave |
|---|---|---|
| [`core`](core/README.md) | Carga de `config.yaml`, logging, conversión a JSON | `Config`, `Logger`, `convert_to_serializable` |
| [`detection`](detection/README.md) | Inferencia YOLO sobre frames de video | `Detector`, `DetectR8` |
| [`tracking`](tracking/README.md) | Orquestación detección → equipo → ByteTrack → tracks | `Tracker`, `ByteTrack` |
| [`identification`](identification/README.md) | Extracción de color de camiseta y asignación de equipo | `ShirtDetector`, `TeamDetector` |
| [`evaluation`](evaluation/README.md) | Métricas cuantitativas por track y experimento | `Evaluator`, `ExperimentVisualizer`, ... |
| [`visualization`](visualization/README.md) | Dibujado de bounding boxes y exportación de video | `Drawer` |

## Flujo de datos del pipeline

```
Video MP4
   │
   ▼
Detector (YOLO)          → detecciones por frame [bbox, conf, clase]
   │
   ▼
TeamDetector (KMeans)    → color de camiseta + asignación de equipo
   │
   ▼
ByteTrack                → IDs persistentes entre frames
   │
   ▼
tracks dict              → {"player": [{id: {bbox, team, ...}}], "ball": [...], ...}
   │
   ├──▶ Drawer           → video MP4 anotado
   └──▶ Evaluator        → métricas JSON / gráficas
```

## Uso básico

```python
from football_ai.core import get_config, Logger, get_logger
from football_ai.tracking import Tracker
from football_ai.visualization import Drawer
from football_ai.evaluation import Evaluator

# 1. Configuración
config = get_config()          # carga config.yaml desde la raíz del proyecto
Logger.setup_from_config(config)
logger = get_logger(__name__)

# 2. Tracking
tracker = Tracker(
    model_path=str(config.get_path('paths', 'models', 'finetuned_player')),
    conf=config.get('detection', 'conf_threshold'),
    tracker_conf=config.tracking,
    team_colors=config.get_team_colors(),
    ball_min_conf=config.get('detection', 'ball_min_conf')
)
tracks = tracker.get_tracks(video_path, show_kmeans=False)

# 3. Visualización
drawer = Drawer(colors=config.get_visualization_colors())
drawer.draw_tracks(tracks, video_path, output_path)

# 4. Evaluación
evaluator = Evaluator()
results = evaluator.evaluate(["player", "ball"], tracks)
```

## Convenciones del paquete

- Todas las rutas se resuelven como absolutas a través de `Config.get_path()`. Nunca se usan rutas hardcodeadas dentro del paquete.
- Los datos de tracking se serializan a JSON con `convert_to_serializable` para manejar tipos NumPy.
- Los colores de OpenCV/BGR se configuran en `config.yaml` para no tener valores de color duplicados en el código.
