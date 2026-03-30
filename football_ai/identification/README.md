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
Asignar cada detección del modelo YOLO al equipo correspondiente a partir del color dominante de la camiseta y, cuando no se usan referencias fijas, aprender automáticamente los colores de equipo mediante clustering.

### Implementación

#### 1. Inicialización y configuración

La implementación actual se configura desde `config.yaml -> color_clustering` y expone estos parámetros principales:

- `n_teams`: número de equipos/clusters a separar.
- `with_ref`: si `true`, reordena los clusters aprendidos usando `team_colors` como referencia externa.
- `team_colors`: colores LAB de referencia opcionales.
- `min_samples`: número mínimo de muestras antes de fijar clusters.
- `min_size_cluster`: tamaño mínimo exigido para cada cluster válido.
- `candidate_classes`: clases que participan en la inferencia de equipo.

Por defecto solo se intentan clasificar detecciones cuya clase esté en `candidate_classes`.

#### 2. Aprendizaje de colores (`update_team_colors`)
Cuando se acumulan al menos `min_samples` colores de camiseta:
1. Se aplica `KMeans` sobre las muestras LAB recogidas.
2. Si aparece un cluster demasiado pequeño (`< min_size_cluster`), se considera ruido y se reintenta sobre el cluster mayor.
3. La referencia final de cada equipo se calcula con la **mediana** de los colores del cluster.
4. Si `with_ref=true`, esas referencias se emparejan con `team_colors`; si no, se nombran de forma neutral (`Equipo 1`, `Equipo 2`, ...).

Hasta que el clustering queda fijado, `detect_teams` puede devolver `team=None` para las detecciones candidatas si no hay suficientes muestras.

#### 3. Asignación (`assign_team`)
Una vez disponibles las referencias, la asignación se hace por **distancia euclidiana mínima en espacio LAB** entre el color detectado y los colores de equipo aprendidos.

#### 4. Extracción de región de camiseta
El crop que se analiza es el **50% superior** del bounding box del jugador. Esto excluye el pantalón, las botas y el césped, que introducían ruido en el clustering.

```python
from football_ai.identification import TeamDetector
import numpy as np

td = TeamDetector(
    n_teams=2,
    with_ref=True,
    team_colors={
        "Real Madrid": np.array([255, 127, 127]),
        "Wolfsburgo":  np.array([224, 77, 196])
    },
    min_samples=60,
    min_size_cluster=4,
    candidate_classes=["player"]
)

# frame_detections es el resultado YOLO de un frame (objeto Results)
team_info_list = td.detect_teams(frame_detections, show_plot=False)
# → lista con un dict por detección:
# [{"class": "player", "team": "Real Madrid", "distances": {...}, "shirt_color": [L, A, B], "bbox_size": float}, ...]
```

#### 5. Visualización de depuración (`visualize_shirt_clusters`)
Función standalone disponible en `football_ai.evaluation.cluster_visualizer` que muestra una cuadrícula con hasta 20 jugadores. Cada jugador ocupa dos columnas: la imagen original del crop y la imagen segmentada por KMeans coloreada con los centroides. Útil para depurar el comportamiento del clustering.

```python
from football_ai.evaluation import visualize_shirt_clusters

visualize_shirt_clusters(tracks, video_path="partido.mp4", team_colors=config.get_team_colors())
