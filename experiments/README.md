# experiments

Notebooks de Jupyter para exploración, análisis y visualización interactiva de los resultados del sistema. Son cuadernos de investigación y no forman parte del pipeline productivo de `scripts/`.

## Estructura

```
experiments/
├── detection/
│   └── finetuning.ipynb              # Análisis del entrenamiento YOLO
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
with open("scripts/tracks.json") as f:
    tracks_todos = json.load(f)

# Comparar experimentos
ev = ExperimentVisualizer(tracks_todos, classes=["player", "ball"])
ev.show_bars_of_experiments("player")
```

Ejecución con Jupyter:
```bash
jupyter lab experiments/
```
