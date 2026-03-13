# identification

Identificación de equipos mediante análisis de color de camiseta. El módulo implementa una cadena de dos pasos: extraer el color dominante de la camiseta de cada jugador y asignarlo al equipo más probable.

---

## `shirt_detector.py` — `ShirtDetector`

### Objetivo
Extraer el color representativo de la camiseta de un jugador a partir del crop de la región superior del bounding box.

### Implementación: KMeans en espacio LAB

**¿Por qué LAB?** El espacio de color CIE L*a*b* es perceptualmente uniforme: la distancia euclidiana entre dos colores LAB se correlaciona mejor con la diferencia perceptual que en RGB o HSV. Esto hace que el clustering sea más estable ante cambios de iluminación en el campo.

**Pasos de `get_color_kmeans(image)`:**
1. Convierte `image` (BGR) a LAB con `cv2.cvtColor`.
2. Aplana la imagen a una lista de píxeles `(N, 3)`.
3. Aplica **KMeans con k=2**: dos clusters, uno para la camiseta y otro para el fondo (césped, zona de piel, publicidad...).
4. Selecciona 7 puntos de referencia en los **bordes y esquinas** del crop (donde estadisticamente aparece más césped y menos camiseta).
5. Predice a qué cluster pertenecen esos puntos de referencia. El cluster más votado es el **fondo**; el otro es la **camiseta**.
6. Devuelve el centroide del cluster de camiseta en espacio LAB.

```python
from football_ai.identification import ShirtDetector

sd = ShirtDetector(n_clusters=2, init='k-means++', n_init=10, random_state=0)

# crop es la región superior del bounding box del jugador (BGR NumPy)
color_lab = sd.get_color_kmeans(crop_bgr)  # → np.array([L, A, B])
```

Los parámetros `n_clusters`, `init`, `n_init` y `random_state` son configurables desde `config.yaml` bajo `color_clustering`.

---

## `team_detector.py` — `TeamDetector`

### Objetivo
Asignar cada detección del modelo YOLO al equipo correspondiente y mantener un estimado del color real de camiseta de cada equipo que se refina progresivamente durante el vídeo.

### Implementación

#### 1. Modos de asignación

`TeamDetector` soporta dos estrategias:

- `reference` (compatibilidad): usa colores de referencia (`teams` en `config.yaml` o `--team-colors`) y ajuste adaptativo.
- `auto-bootstrap`: ignora nombres de equipo predefinidos, agrupa colores de camiseta en los frames iniciales y fija una referencia robusta por equipo para todo el vídeo (nombres neutrales, p. ej. `Equipo 1`, `Equipo 2`).

En ambos modos, por defecto solo participan `player` y `goalkeeper` en la inferencia de equipo.

#### 2. Sistema de confirmación adaptativo (`update_team_colors`)
Antes de que un equipo tenga suficientes muestras, el color de referencia puede ser impreciso. El sistema:
1. Extrae el color de camiseta del jugador con `ShirtDetector`.
2. Calcula la distancia al equipo más cercano.
3. Agrupa muestras similares (distancia < `color_tolerance` en espacio LAB).
4. Cuando un grupo acumula ≥ `confirmation_threshold` detecciones, **actualiza el color de referencia del equipo** al centroide de ese grupo.
5. Ese equipo queda marcado como "confirmado" y su color ya no se actualiza más.

Esto permite que el sistema se adapte automáticamente al color exacto de las camisetas en las condiciones de iluminación del partido, en lugar de depender únicamente de los colores precalibrados.
En `auto-bootstrap`, tras cerrar bootstrap, la referencia de cada equipo se calcula como **mediana por cluster** y queda fija para evitar cambios de etiqueta durante el vídeo. Además, se exige un mínimo de muestras por cluster (por defecto 4): si aparece un cluster pequeño (<=3), se re-clusteriza sobre el cluster grande para evitar que outliers (p. ej. árbitros) dominen un equipo.

#### 3. Asignación (`assign_team`)
Con los colores (iniciales o confirmados), asigna el equipo por **distancia euclidiana mínima en espacio RGB** entre el color de camiseta detectado y los colores de referencia actualizados.

#### 4. Extracción de región de camiseta (`get_team_of_players`)
El crop que se analiza es el **50% superior** del bounding box del jugador. Esto excluye el pantalón, las botas y el césped, que introducían ruido en el clustering.

```python
from football_ai.identification import TeamDetector
import numpy as np

td = TeamDetector(
    team_colors_refs={
        "Real Madrid": np.array([255, 127, 127]),
        "Wolfsburgo":  np.array([224, 77, 196])
    },
    confirmation_threshold=3,   # de config.yaml: color_clustering.confirmation_threshold
    color_tolerance=25,         # de config.yaml: color_clustering.color_tolerance
    assignment_mode="auto-bootstrap",
    auto_bootstrap_frames=1,
    auto_bootstrap_min_samples=12
)

# frame_detections es el resultado YOLO de un frame (objeto Results)
team_info_list = td.detect_teams(frame_detections, show_plot=False)
# → lista con un dict por detección:
# [{"class": "player", "team": "Real Madrid", "distances": {...}, "shirt_color": np.array, "bbox_size": int}, ...]
```

#### 5. Visualización de depuración (`visualize_shirt_clusters`)
Función standalone disponible en `football_ai.evaluation.cluster_visualizer` que muestra una cuadrícula con hasta 20 jugadores. Cada jugador ocupa dos columnas: la imagen original del crop y la imagen segmentada por KMeans coloreada con los centroides. Útil para depurar el comportamiento del clustering.

```python
from football_ai.evaluation import visualize_shirt_clusters

visualize_shirt_clusters(tracks, video_path="partido.mp4", team_colors=config.get_team_colors())
