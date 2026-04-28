# experiments

Notebooks de Jupyter para exploración, análisis y visualización interactiva de los resultados del sistema. Son cuadernos de investigación y no forman parte del pipeline productivo de `scripts/`.

## Estructura

```
experiments/
├── detection/
│   └── finetuning.ipynb              # Análisis del entrenamiento YOLO
├── possession/
│   ├── README.md                     # Experimento de posesión por equipo
│   └── team_possession.py            # Heurística de posesión + render anotado
├── set_transformer.ipynb             # Entrenamiento + aplicación de Set Transformer para roles
├── positions/
│   ├── position_dataset.py            # Reexport a football_ai.positions.data.dataset
│   ├── set_transformer_pipeline.py    # Reexport a football_ai.positions.model.train
│   └── position_role_dataset.ipynb    # Pipeline guiado de etiquetado y exportación del dataset
├── reference_points/
│   ├── classical_reference_points.py  # Utilidades para homografía clásica
│   ├── pnlcalib_reference_points.py   # Wrapper para inferencia con PnLCalib
│   ├── pnlcalib_reference_points.ipynb# Homografía y bird-eye con PnLCalib
│   ├── reference_points.ipynb         # Detección automática de puntos del campo
│   ├── soccana_keypoints.py           # Utilidades para el modelo Soccana_Keypoint
│   └── soccana_keypoints.ipynb        # Homografía basada en keypoints detectados por IA
├── tracking/
│   └── balon.ipynb                   # Depuración del tracking del balón
└── visualization/
    ├── clusters_colores.ipynb         # Calibración de colores de equipo
    ├── experiments_comparator.ipynb   # Comparador de experimentos de tracking
    ├── team_detector_relabel_flow.ipynb # Flujo YOLO -> homografía -> TeamDetector con relabel debug
    ├── team_detector_relabel_flow_utils.py # Helper para exportar y graficar artefactos del relabel
    └── track_evolution.ipynb          # Análisis de tracks individuales
```

---

## `detection/finetuning.ipynb`

**Objetivo:** Analizar los resultados del fine-tuning de YOLO: curvas de pérdida de entrenamiento y validación, métricas de precisión/recall por clase, muestras de predicciones del modelo.

Sirve para decidir cuándo parar el entrenamiento y comparar versiones de modelo.

---

## `tracking/balon.ipynb`

**Objetivo:** Depurar y validar el tracking del balón, que es el objeto más problemático por su tamaño pequeño y velocidad alta.

Puntos de análisis:
- Cuántos frames activa el tracker vs. los que usa el fallback de detección cruda.
- Distribución de confianzas de detección del balón.
- Visualización frame a frame de los tracks generados.
- Ajuste del parámetro `ball_min_conf` para el fallback.

---

## `possession/team_possession.py`

**Objetivo:** inferir por frame qué equipo tiene la posesión del balón usando únicamente tracks de `ball`, `player` y `goalkeeper`, y renderizar el equipo poseedor en una esquina del vídeo.

La heurística combina:
- filtrado temporal del balón con posición esperada según la trayectoria reciente;
- distancia del balón al pie del jugador más cercano;
- cambios de velocidad o dirección del balón para detectar toque/control;
- confirmación temporal antes de aceptar muchos cambios rivales;
- estabilización corta de lagunas/segmentos para evitar parpadeos en el vídeo;
- continuidad temporal de la posesión para cubrir pases en tránsito dentro del mismo equipo;
- persistencia del último equipo que tocó el balón, sin liberar la posesión a `unknown` salvo antes del primer toque detectado.

Entrada típica:
- vídeo en `data/partidoPrueba/`
- tracks en `output/tracks_json/tracker/<video>_tracks.json`

Salidas:
- `output/predictions/possession/<video>_<timestamp>/frame_possession.csv`
- `output/predictions/possession/<video>_<timestamp>/summary.json`
- `output/predictions/possession/<video>_<timestamp>/<video>_possession_annotated.mp4`

Ejecución:

```bash
python -m experiments.possession.team_possession \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```

---

## `positions/position_role_dataset.ipynb`

**Objetivo:** Construir un dataset supervisado para clasificar el rol nominal de cada jugador (`POR`, `LI`, `DFC_IZQ`, `MC`, `DC`, etc.) usando tracks ya proyectados al campo.

Pipeline que implementa:
- Carga tracks desde `output/tracks_json/tracker/` y construye observaciones por frame (`x,y` normalizados, `x_m,y_m`, equipo, bbox, confianza).
- Selecciona un frame inicial con suficientes jugadores visibles para etiquetar IDs.
- Permite definir `ROLE_MAP` manual (`team_id -> player_id -> role_label`).
- Propaga etiquetas al resto de frames por ID canónico.
- Reorienta coordenadas por equipo para que el ataque del equipo objetivo apunte a `+x`.
- Construye muestras con features del jugador objetivo + tensor de compañeros (con padding y máscara).
- Exporta el dataset a `data/posiciones_etiquetadas/<match_id>_<timestamp>/`.

Salidas principales:
- `base_table.csv`
- `samples_metadata_and_obj_features.csv`
- `samples_teammates.npz`
- `dataset_meta.json`

Requisitos:
- Vídeos en `data/partidosPosiciones/`
- `tracks.json` o `<video>_tracks.json` con `field_position_m` en tracks de `player/goalkeeper`

---

## `positions/position_dataset.py`

**Objetivo:** Wrapper fino hacia `football_ai.positions.data.dataset` para notebooks y scripts experimentales.

Funciones clave:
- `list_position_videos`: enumera vídeos en `data/partidosPosiciones/`.
- `resolve_tracks_path_for_video`: busca `output/tracks_json/tracker/<video_sanitizado>_tracks.json` y cae a `tracks.json` legacy.
- `build_observations_from_tracks`: convierte tracks a tabla tabular por jugador/frame.
- `add_velocity_features`: añade `vx, vy` por jugador.
- `choose_label_frame`: selecciona frame recomendado para etiquetado manual.
- `render_frame_with_player_ids`: renderiza preview con `player_id:team_id`.
- `validate_role_map` y `apply_role_map`: validación y aplicación del etiquetado manual.
- `infer_attack_direction_by_team`: estima sentido de ataque por equipo.
- `build_role_samples`: crea samples y tensores de compañeros + máscara para modelado.

Labels permitidas (v1): `POR, LI, DFC_IZQ, DFC_DER, LD, MC, MI, MD, EI, ED, DC`.

---

## `set_transformer.ipynb`

**Objetivo:** usar un checkpoint ya entrenado de Set Transformer, o reentrenarlo si hace falta, y aplicarlo sobre `data/partidoPrueba/partido_ajustado.mp4`.
La primera celda recarga explícitamente `experiments.positions.set_transformer_pipeline` para evitar que Jupyter reutilice una versión antigua del módulo tras editar el `.py`.

Pipeline que implementa:
- reconstruye el dataset de entrenamiento a partir de `data/posiciones_etiquetadas/common/base_table.csv`;
- embebe el jugador objetivo con una MLP pequeña;
- embebe el set de compañeros con otra MLP;
- canoniza el campo por equipo con una rotación de 180° cuando el ataque va hacia `-x`, para preservar la semántica `IZQ/DER`;
- resume el contexto colectivo con Set Transformer;
- permite imponer el once esperado por equipo con Hungarian sobre las probabilidades agregadas por jugador, dejando plazas vacías si falta detección y reasignando sobrantes al mejor rol permitido;
- fusiona `[h_obj; h_set]` y clasifica el rol final;
- exporta checkpoint, métricas, predicciones por frame, resumen estable por jugador y opcionalmente el vídeo anotado.

Artefactos principales:
- `models/positions/set_transformer/<timestamp>/set_transformer_checkpoint.pt`
- `models/positions/set_transformer/<timestamp>/metrics.json`
- `output/predictions/positions/partido_ajustado_<timestamp>/frame_role_predictions.csv`
- `output/predictions/positions/partido_ajustado_<timestamp>/player_role_summary.csv`
- `output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json`

---

## `reference_points/reference_points.ipynb`

**Objetivo:** Estimar automáticamente una homografía entre el frame broadcast y una plantilla 2D del campo para poder proyectar jugadores a coordenadas métricas del terreno de juego.

Implementa un pipeline clásico de visión:
- segmentación del césped mediante umbrales HSV;
- extracción de líneas blancas con combinación de brillo, baja saturación, top-hat y filtros para eliminar blobs compactos de jugadores;
- detección de segmentos con Hough;
- clustering en dos familias de orientación;
- intersecciones candidatas como puntos de referencia;
- construcción de rectángulos candidatos a partir de pares de líneas;
- búsqueda y scoring de homografías contra una plantilla métrica del campo;
- procesamiento secuencial frame a frame con reutilización temporal de la homografía previa.

El notebook también incluye una celda opcional para reutilizar `output/tracks_json/tracker/tracks.json` y proyectar al bird-eye los puntos pie de los tracks ya generados.

---

## `reference_points/soccana_keypoints.ipynb`

**Objetivo:** Estimar la homografía del campo usando el modelo de Hugging Face `Adit-jain/Soccana_Keypoint`, que devuelve 29 keypoints semánticos del terreno de juego.

Implementa un pipeline basado en modelo:
- descarga del peso `best.pt` desde Hugging Face si no existe localmente;
- inferencia pose con Ultralytics YOLO sobre el frame broadcast;
- mapeo de los 29 keypoints a coordenadas métricas del campo;
- estimación de homografía con `cv2.findHomography(..., RANSAC)`;
- procesado secuencial frame a frame con suavizado temporal ligero;
- proyección opcional de los tracks ya existentes al bird-eye.

Por defecto, el cuaderno recorre los frames iniciales y busca el primer frame donde la homografía proyecta suficientes jugadores dentro del campo 2D, para descartar transformaciones geométricamente válidas pero visualmente erróneas.

Es un experimento alternativo al enfoque clásico, no un reemplazo del notebook anterior.

---

## `reference_points/pnlcalib_reference_points.ipynb`

**Objetivo:** Probar `PnLCalib` como tercer enfoque para estimar la homografía del campo y generar una vista bird-eye a partir de un frame broadcast.

Implementa un wrapper ligero sobre el repositorio externo:
- clona `PnLCalib` en `models/reference_points/pnlcalib_repo/` si no existe;
- descarga los pesos `SV_kp` y `SV_lines` de GitHub Releases;
- carga los modelos HRNet del propio repositorio;
- detecta keypoints y líneas del campo en cada frame;
- obtiene la homografía `imagen -> campo` y genera el warp bird-eye;
- proyecta opcionalmente los tracks del `tracks.json` ya generado para validar si la geometría es coherente;
- reescala los puntos de track a la resolución real del frame usado por la homografía y los puede pintar directamente sobre el mismo bird-eye transformado.

Por defecto recorre frame a frame desde el inicio del vídeo hasta el frame `100` y busca el primer frame donde haya suficientes personas proyectadas dentro del campo 2D.

Importante: `PnLCalib` está licenciado como `GPL-2.0`. Este cuaderno se deja como experimento de investigación; si este método se integrase en producto habría que revisar esa licencia antes.

---

## `visualization/clusters_colores.ipynb`

**Objetivo:** Calibrar los colores de referencia de los equipos y validar que el `TeamDetector` asigna correctamente cada jugador a su equipo.

Usa `ClusterVisualizer` para:
- Mostrar histogramas de la distribución de colores LAB detectados por clase y equipo.
- Visualizar los clusters KMeans sobre crops reales de camisetas.
- Identificar colores de referencia óptimos para `config.yaml`.

---

## `visualization/experiments_comparator.ipynb`

**Objetivo:** Comparar cuantitativamente múltiples ejecuciones de `track_experiments.py` con distintas combinaciones de hiperparámetros del tracker.

Usa `ExperimentVisualizer` para:
- Cargar el fichero `tracks.json` generado por `track_experiments.py`.
- Generar gráficas de barras comparando métricas (coverage, fragmentación, flip rate...) por experimento.
- Boxplots de la distribución de métricas por track dentro de cada experimento.
- Seleccionar la mejor combinación de hiperparámetros para `config.yaml`.

---

## `visualization/team_detector_relabel_flow.ipynb`

**Objetivo:** entender visualmente por qué `TeamDetector` relabela ciertas detecciones (`player -> referee` o `player -> goalkeeper`) siguiendo el flujo real de producción:

1. YOLO sobre `video_prueba_medio`.
2. Homografía con `PnLCalibFieldProjector`.
3. `TeamDetector` con el mismo estado incremental que usa el tracker.

El notebook guarda artefactos en `output/analysis/team_detector_<video>/`:
- `detections.csv`: una fila por detección con bbox, confianza, color LAB, `new_possible_class`, clase final, distancias a cada referencia y `upper_bound` por equipo.
- `frames.jsonl`: estado por frame del bootstrap de colores, diagnóstico de homografía, referencias activas y stats dinámicas de color.
- `summary.json`: resumen agregado.
- `bootstrap_samples.json` y `bootstrap_samples/`: primeras muestras realmente aceptadas por la lógica actual para cerrar el bootstrap inicial de `player` y `referee` (ya filtradas por `confidence`), con crops guardados para inspección visual.

Visualizaciones incluidas:
- scatter 3D en espacio LAB de todas las camisetas detectadas, coloreando cada punto/cruz con el color real detectado de esa camiseta;
- dos figuras de bootstrap (`player` y `referee`) con 8 muestras por fila; en cada muestra se ve el crop real y, al lado, el crop recoloreado con los means predichos por KMeans;
- trayectoria temporal de las referencias LAB de cada clase y snapshots muestreados de las esferas `upper_bound` de los equipos de campo;
- mapa del campo con posiciones métricas y detecciones relabeleadas;
- timeline de relabels por frame;
- histograma del margen `distancia - upper_bound` para ver qué detecciones quedan fuera del cluster;
- visor frame a frame con bbox y etiqueta `yolo -> final`.

Es especialmente útil para responder preguntas del tipo:
- “¿Estas detecciones salen de `player` porque realmente están fuera del cluster de color?”
- “¿Se van a `referee` por color o por gate posicional?”
- “¿En qué frames el bootstrap aprende referencias demasiado estrechas?”

---

## `visualization/track_evolution.ipynb`

**Objetivo:** Analizar en detalle la evolución temporal de tracks individuales para entender problemas concretos (ID switches, gaps, confusión de equipo).

Usa `TrackVisualizer` y `MetricsVisualizer` para:
- Histogramas de todas las métricas de una clase.
- Scatter interactivo de velocidad vs. frame.
- Identificar tracks problemáticos por su alta tasa de flips o baja cobertura.

---

## Cómo usar los notebooks

Los notebooks cargan datos desde los JSON generados por los scripts:

```python
import json
from football_ai.evaluation import Evaluator, ExperimentVisualizer

# Cargar tracks de una ejecución
with open("output/pruebaTracker/tracks.json") as f:
    tracks_todos = json.load(f)

# Comparar experimentos
ev = ExperimentVisualizer(tracks_todos, classes=["player", "ball"])
ev.show_bars_of_experiments("player")
```

Ejecución con Jupyter:
```bash
jupyter lab experiments/
```
