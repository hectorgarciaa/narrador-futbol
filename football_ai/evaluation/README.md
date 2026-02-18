# evaluation

Módulo de análisis cuantitativo y visual de los resultados del tracking. Permite medir la calidad del tracker, comparar distintas configuraciones de hiperparámetros e identificar problemas concretos (ID switches, fragmentación, confusión de equipo...).

---

## `evaluator.py` — `Evaluator`

### Objetivo
Calcular métricas cuantitativas por cada `track_id` y resumir el rendimiento global del tracker para una clase de objeto.

### ¿Cómo funciona?

`evaluate(classes, tracks)` itera sobre cada clase y llama a `evaluateClass(tracks, class_name)`, que:
1. Recorre todos los frames de esa clase y agrupa la información por `track_id` (usando `defaultdict`).
2. Para cada track, calcula las métricas con `metricsOfTic()`.
3. Genera un `summary` global agregando las métricas de todos los tracks.

### Métricas por track (`metricsOfTic`)

| Métrica | Descripción | Cómo se calcula |
|---|---|---|
| `coverage` | % de frames del vídeo en que el track fue visible | `frames_vistos / total_frames` |
| `fragments` | Número de interrupciones en el track | Gaps entre frames no consecutivos |
| `mean_gap_len` | Longitud media de los saltos en frames | Media de la duración de cada interrupción |
| `mean_speed` | Velocidad media de movimiento | Distancia euclidiana entre bboxes consecutivos / frames |
| `max_speed` | Velocidad máxima | Máximo de los samples de velocidad |
| `team_flip_rate_static` | % de frames con equipo distinto al mayoritario | Frames anómalos / total frames |
| `team_flip_rate_dynamic` | % de cambios consecutivos de equipo | Transiciones / (total - 1) |
| `entropy` | Entropía de la distribución de equipos | $-\sum p_i \log_2 p_i$ |
| `color_var` | Varianza del color de camiseta a lo largo del tiempo | Media de la varianza por canal LAB |
| `color_diff` | Variación de color entre frames consecutivos | Media de la norma de la diferencia |
| `bbox_size_cv` | Coeficiente de variación del tamaño del bbox | `std / mean` del área |
| `mean_confidence` | Confianza media de las detecciones del track | Media de los scores YOLO |

### Formato de salida

```python
evaluator = Evaluator()
results = evaluator.evaluate(["player", "goalkeeper", "referee", "ball"], tracks)

# results["player"]["summary"]  → dict con métricas globales
# results["player"]["metrics"]  → {track_id: {métrica: [{id, frame, value}]}}
# results["player"]["n_frames"] → total de frames del vídeo
```

Cada métrica se almacena como lista de **eventos** `{"id", "frame", "value"}` para poder graficarlos temporalmente.

---

## `experiment_visualizer.py` — `ExperimentVisualizer`

### Objetivo
Comparar múltiples ejecuciones del tracker con distintos hiperparámetros (grid search manual) mediante tablas y gráficas.

### Uso
```python
ev = ExperimentVisualizer(
    experimentos_tracks=tracks_todos,  # lista de {"conf", "mcf", "mt", "tt", "track"}
    classes=["player", "ball"]
)

# Resumen global por experimento
df_summary = ev.createDFSummary("player")

# Métricas detalladas por track y experimento
df_metrics  = ev.createDfsMetrics("player")

# Gráficas de barras: una por métrica, eje X = experimento
ev.shorBarsOfExperiments("player")

# Boxplot de una métrica concreta
ev.showBoxplotOfExperiments("player", y="coverage")
```

---

## `metrics_visualizer.py` — `MetricsVisualizer`

### Objetivo
Generar gráficas interactivas Plotly de métricas temporales para analizar el comportamiento del tracker frame a frame.

Usa el formato de eventos (listas de `{id, frame, value}`) que devuelve `Evaluator`.

```python
mv = MetricsVisualizer()

# Velocidad de todos los tracks a lo largo del vídeo
speed_events = mv.collect_speed_events(metrics)
mv.plot_speed_events_scatter(speed_events)   # scatter frame vs speed
mv.plot_speed_histogram(speed_events)        # histograma de velocidades

# Cualquier otra métrica (coverage, color_var, bbox_size_cv...)
events = mv.collect_metric_events(metrics, "coverage")
mv.plot_metric_events_scatter(events, "coverage")
mv.plot_metric_histogram(events, "coverage")
```

---

## `track_visualizer.py` — `TrackVisualizer`

### Objetivo
Analizar la distribución de métricas para todos los tracks de una clase concreta en un experimento concreto mediante histogramas.

```python
tv = TrackVisualizer(tracks=tracks, class_name="player")
df = tv.getDfMetricsList()        # DataFrame con todas las métricas por track
tv.showAllHistsMetricsList()      # cuadrícula de histogramas
tv.showHist(df["coverage"], "coverage", bins=30)
```

Internamente llama a `Evaluator.evaluateClass()` en el constructor, por lo que no requiere pasar métricas precomputadas.

---

## `cluster_visualizer.py` — `ClusterVisualizer`

### Objetivo
Visualizar la distribución de colores de camiseta detectados por el `TeamDetector` para calibrar y validar el clustering de equipos.

```python
cv = ClusterVisualizer(tracks=tracks, video_path="partido.mp4")

# Acceder al DataFrame
df = cv.getDataFrame()   # columnas: class_name, frame, tracker_id, x1, y1, w, h,
                         #           bbox_size, confidence, Real Madrid, Wolfsburgo, L, A, B, team

# Histogramas de distribución de colores
cv.showHist(cols=["L", "A", "B"], xlabel="Color LAB", width=10)
```

Construye internamente un DataFrame plano de todas las detecciones con sus colores LAB, lo que permite hacer filtros y análisis ad hoc con pandas.

> **Limitación conocida:** `ClusterVisualizer` y `reformatTrack` tienen los nombres de equipo "Real Madrid" y "Wolfsburgo" hardcodeados en las columnas del DataFrame. Si se cambian los equipos en `config.yaml`, hay que actualizar este método.
