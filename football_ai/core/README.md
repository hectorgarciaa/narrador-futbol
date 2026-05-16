# core

Módulo de infraestructura transversal del proyecto. Proporciona las tres utilidades que necesitan todos los demás módulos: gestión de configuración, logging centralizado y serialización de datos.

## Archivos

### `config.py` — `Config` y `get_config`

**Objetivo:** Proporcionar un punto único de acceso a todos los parámetros del proyecto, leyéndolos de `config.yaml` y evitando que haya valores hardcodeados dispersos por el código.

**Funcionamiento:**
1. `Config.from_yaml()` busca automáticamente `config.yaml` subiendo desde la ubicación del archivo hasta la raíz del proyecto.
2. Carga el YAML con `yaml.safe_load` y almacena el diccionario internamente.
3. Resuelve rutas relativas a absolutas usando `project_root / ruta_relativa`, lo que permite ejecutar scripts desde cualquier directorio.
4. `get_config()` implementa el patrón **singleton**: carga la config una vez y la reutiliza en todas las llamadas posteriores dentro de la misma ejecución.

**API principal:**

```python
from football_ai.core import get_config

config = get_config()   # singleton – carga config.yaml automáticamente

# Rutas absolutas (crea el directorio si create_if_missing=True)
model_path = config.get_path('paths', 'models', 'finetuned_player')
output_dir = config.get_path('paths', 'output', 'prueba_tracker', create_if_missing=True)

# Valores escalares
conf_thresh = config.get('detection', 'conf_threshold')   # → 0.01
epochs      = config.get('finetuning', 'epochs')          # → 50

# Propiedades agrupadas
tracking_cfg   = config.tracking     # → orquestacion general (execution_mode, lineup_spec, runtime)
projector_cfg  = config.projector    # → calibracion / homografia
bytetracker_cfg = config.bytetracker # → asociacion ByteTrack
canonical_cfg  = config.canonical    # → canonizacion y logica de balon
positions_cfg  = config.positions    # → roles / estabilizacion posicional
teams_cfg      = config.teams        # → dict de equipos con colores

# Helpers de alto nivel
team_colors  = config.get_team_colors()          # → usa `color_lab_opencv` (o `color_rgb` legacy)
bgr_colors   = config.get_visualization_colors() # → {"player": (0,255,0), ...} en BGR para OpenCV
```

**Propiedades disponibles:** `paths`, `detection`, `tracking`, `teams`, `visualization`, `finetuning`, `logging_config`.

**Manejo de errores:**
- `FileNotFoundError` si no encuentra `config.yaml`.
- `KeyError` con mensaje descriptivo si se accede a una clave inexistente.
- `ValueError` si el valor en la clave pedida no es una ruta válida (string).

---

### `logger.py` — `Logger` y `get_logger`

**Objetivo:** Configurar el sistema de logging del proyecto una sola vez desde la config, y proporcionar loggers nombrados por módulo para trazabilidad.

**Funcionamiento:**
- Usa el módulo estándar `logging` de Python.
- Implementa el patrón **inicialización única** con `_initialized`: si ya se configuró, las llamadas posteriores a `setup()` no hacen nada.
- Soporta dos destinos simultáneamente: consola (`stdout`) y archivo de log.
- Los loggers se cachean en `_loggers` para no crear instancias duplicadas.

```python
from football_ai.core import Logger, get_logger

# Inicializar una vez al arrancar el script
Logger.setup_from_config(config)  # lee nivel, formato y ruta de log de config.yaml

# En cualquier módulo
logger = get_logger(__name__)  # nombre = "football_ai.tracking.tracker", etc.
logger.debug("Detalle interno")
logger.info("Frame procesado: 42")
logger.warning("Track sin equipo asignado")
logger.error("No se pudo abrir el video", exc_info=True)
```

El formato por defecto es:
```
2026-02-18 12:34:56 - football_ai.tracking.tracker - INFO - Frame procesado: 42
```

---

### `serialization.py` — `convert_to_serializable`

**Objetivo:** Convertir recursivamente estructuras de datos que contienen tipos NumPy (arrays, integers, floats) a tipos nativos de Python para que `json.dump` no falle.

**Funcionamiento:** Recorre recursivamente dicts, listas y arrays, convirtiendo:
- `np.ndarray` → `list`
- `np.integer` → `int`
- `np.floating` → `float`
- Cualquier otro tipo → se devuelve tal cual

```python
from football_ai.core import convert_to_serializable
import json

# tracks contiene np.arrays en bbox, shirt_color, distances...
tracks_serializable = convert_to_serializable(tracks)

with open("tracks.json", "w", encoding="utf-8") as f:
    json.dump(tracks_serializable, f, indent=4, ensure_ascii=False)
```
