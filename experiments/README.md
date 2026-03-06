# experiments

Notebooks de Jupyter para exploración, análisis y visualización interactiva de los resultados del sistema. Son cuadernos de investigación y no forman parte del pipeline productivo de `scripts/`.

## Estructura

```
experiments/
├── detection/
│   └── finetuning.ipynb              # Análisis del entrenamiento YOLO
├── positions/
│   ├── __init__.py
│   ├── position_dataset.py            # Utilidades para construir dataset de roles
│   └── position_role_dataset.ipynb    # Etiquetado por ID y export de dataset posicional
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

## `positions/position_role_dataset.ipynb`

**Objetivo:** Construir un dataset supervisado para clasificación de rol nominal (`POR`, `LI`, `DFC_IZQ`, etc.) a partir de tracks proyectados al campo 2D.

Flujo del cuaderno:
- Carga `output/tracks_json/tracker/<nombre_video>_tracks.json` (con fallback legacy a `tracks.json`) y genera tabla base con `match_id`, `frame_id`, `team_id`, `player_id`, `x`, `y`, `visible`.
- Muestra un frame de inicio con IDs dibujados para etiquetar manualmente `player_id -> role_label`.
- Extiende la etiqueta a todos los frames por ID.
- Infere (u opcionalmente fija) dirección de ataque por equipo y reorienta cada muestra para que el equipo objetivo ataque hacia `+x`.
- Construye features del jugador objetivo y de compañeros, incluyendo tensor con `padding` y `mask`.
- Exporta dataset a `output/datasets/positions/<match_id>_<run_id>/`.

`position_dataset.py` contiene las utilidades reutilizables del notebook (parseo de tracks, validación de etiquetas, inferencia de orientación y construcción de muestras).

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
