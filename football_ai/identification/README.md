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

La implementación actual combina dos fuentes de configuración:

- `tracking.team_assignment_mode`: `reference` o `auto-bootstrap`.
- `tracking.team_bootstrap_*`: controla cuántos frames/muestras se usan para fijar los clusters en modo bootstrap.
- `tracking.team_candidate_classes`: clases que aportan muestras de color.
- `color_clustering.*`: hiperparámetros del `ShirtDetector` y compatibilidad con la configuración antigua.
- `teams` / `--team-colors` / `lineup_spec.json`: referencias LAB opcionales para nombrar equipos.

En ambos modos, por defecto participan `player` y `goalkeeper` en la inferencia de equipo.

#### 2. Modo `reference`
Cuando `team_assignment_mode=reference`, `TeamDetector` arranca con colores LAB de referencia y asigna cada detección al equipo más cercano. Además, mantiene un pequeño mecanismo de confirmación:

1. Busca el equipo de referencia más cercano al color detectado.
2. Acumula muestras parecidas por equipo.
3. Cuando una muestra se repite `confirmation_threshold` veces dentro de `color_tolerance`, sustituye la referencia por ese color confirmado del vídeo.

Así el detector empieza funcionando desde el primer frame, pero puede adaptarse ligeramente a la iluminación real del partido.

#### 3. Modo `auto-bootstrap`
Cuando `team_assignment_mode=auto-bootstrap`:

1. Se recogen colores de camiseta durante los primeros frames.
2. Se aplica `KMeans` para separar `auto_num_teams` clusters.
3. Si un cluster es demasiado pequeño (`auto_min_cluster_samples`), se trata como ruido y se reintenta sobre el cluster mayor.
4. La referencia final de cada equipo se calcula con la **mediana** LAB de cada cluster.

Si además se usa un `lineup_spec.json` generado por la interfaz, el bootstrap sigue aprendiendo los centroides reales del vídeo, pero renombra los clusters según los colores de equipo introducidos por el usuario.

Hasta que el bootstrap queda fijado, `detect_teams` puede devolver `team=None` para las detecciones candidatas.

#### 4. Asignación (`assign_team`)
Una vez disponibles las referencias activas, la asignación se hace por **distancia euclidiana mínima en espacio LAB** entre el color detectado y los colores de equipo aprendidos o confirmados.

#### 5. Extracción de región de camiseta
El crop que se analiza es el **50% superior** del bounding box del jugador. Esto excluye el pantalón, las botas y el césped, que introducían ruido en el clustering.

```python
from football_ai.identification import TeamDetector
import numpy as np

td = TeamDetector(
    team_colors_refs={
        "Real Madrid": np.array([255, 127, 127]),
        "Wolfsburgo":  np.array([224, 77, 196])
    },
    assignment_mode="auto-bootstrap",
    auto_bootstrap_min_samples=12,
    auto_min_cluster_samples=4,
    team_candidate_classes=["player", "goalkeeper"],
)

# frame_detections es el resultado YOLO de un frame (objeto Results)
team_info_list = td.detect_teams(frame_detections, show_plot=False)
# → lista con un dict por detección:
# [{"class": "player", "team": "Real Madrid", "distances": {...}, "shirt_color": [L, A, B], "bbox_size": float}, ...]
```

#### 6. Visualización de depuración (`visualize_shirt_clusters`)
Función standalone disponible en `football_ai.evaluation.cluster_visualizer` que muestra una cuadrícula con hasta 20 jugadores. Cada jugador ocupa dos columnas: la imagen original del crop y la imagen segmentada por KMeans coloreada con los centroides. Útil para depurar el comportamiento del clustering.

```python
from football_ai.evaluation import visualize_shirt_clusters

visualize_shirt_clusters(tracks, video_path="partido.mp4", team_colors=config.get_team_colors())
