# identification

Cuarta fase del pipeline de tracking.  
Consume `FILTERING` y emite `IDENTIFICATION`.

## Objetivo

Enriquecer cada detección con señales de color y clase relabelada para tracking:

- `class_td`: clase final usada aguas abajo.
- `team`: equipo asignado si aplica.
- `shirt_color`: color LAB representativo de la camiseta.
- `distances`: distancias LAB a equipos/clusters activos.
- `bbox_size`: área del bbox.

## Entrada esperada

`filtering_packet["clean"]`:

```python
{
    "num_detections": int,
    "det_id": list[int],
    "bbox_xyxy": list[list[float]],
    "confidence": list[float],
    "class_name": list[str],
    "field_positions_m": list[list[float]],
    "ground_points_image_original": list[list[float]],
    "homography_valid": bool,  # opcional, propagado por filtering si existe
    "field_positions_usable_for_tracking": bool,  # opcional, propagado por filtering si existe
}
```

Las clases de entrada ya están normalizadas: `player`, `goalkeeper`, `referee`, `ball`.

## Salida (`IDENTIFICATION`)

`clean` mantiene los campos de entrada consumidos después y añade:

```python
{
    "num_detections": int,
    "det_id": list[int],
    "bbox_xyxy": list[list[float]],
    "confidence": list[float],
    "class_name": list[str],
    "field_positions_m": list[list[float]],
    "ground_points_image_original": list[list[float]],
    "class_td": list[str],
    "team": list[str | None],
    "shirt_color": list[list[float] | None],
    "distances": list[dict | None],
    "bbox_size": list[float],
}
```

## Flujo real del módulo

1. `TeamDetector.identify_packet(...)` consume el packet `FILTERING`.
2. `ShirtDetector` extrae un crop de camiseta a partir del 50% superior del bbox.
3. El crop se redimensiona, convierte a LAB y se clusteriza con `BatchedKMeansGPU`.
4. Se estima un color de camiseta por detección candidata (`player`, `goalkeeper`, `referee`).
5. `TeamColorModel`:
   - mantiene/actualiza clusters de equipos de campo,
   - mantiene/actualiza cluster de árbitro,
   - devuelve asignación de equipo y distancias.
6. `TeamDetector` aplica reglas de relabel:
   - `player -> referee` si color y gate posicional lo permiten,
   - `player/referee -> goalkeeper` si es outlier de campo y el gate posicional lo permite,
   - o mantiene la clase actual si no hay evidencia suficiente.

## Posición de campo y gates

`IDENTIFICATION` mantiene dos semánticas separadas:

En la práctica:

- si `field_positions_m[i]` llega como coordenada finita, se conserva en el packet y puede seguir informando otras fases;
- si llega como `None` o con valores no finitos, esa detección cae a `field_position=None`;
- los gates posicionales y los relabels que dependen de `field_position` solo se activan cuando `homography_valid == True` y `field_positions_usable_for_tracking == True`;
- si la homografía no es usable, la fase sigue usando color y distancias, pero desactiva las decisiones de clase que dependan de la posición proyectada.

## `ShirtDetector`

API real:

```python
detector = ShirtDetector(
    pixels_resize=1536,
    batched_kmeans_gpu_conf={
        "n_clusters": 2,
        "use_gpu": True,
    },
)

colors = detector.get_color_kmeans_batch(list_of_crops_bgr)
```

Notas importantes:

- El algoritmo actual solo admite `n_clusters == 2`.
- Un crop o clusterizacion invalida devuelve `None`, no un negro sintético `[0, 0, 0]`.
- Si la clusterización existe pero todos los puntos de referencia caen en un único label, se conserva el fallback histórico y se usa el cluster opuesto como color de camiseta en vez de anular la muestra.
- Los bboxes se clipean al frame antes de extraer crop.

## `TeamColorModel`

Responsabilidades:

- bootstrap y actualización de clusters de `player`,
- bootstrap y actualización de cluster `referee`,
- cálculo de distancias LAB por equipo,
- reglas robustas para decidir si un color cae dentro de cluster o es outlier.

Decisión actual sobre `goalkeeper`:

- `goalkeeper` participa en extracción de color y en asignación/relabel,
- pero no aporta muestras al bootstrap de clusters.

Esto es intencional para no contaminar los clusters de equipos de campo con camisetas de portero.

## Runtime vs Debug

- `execution_mode="runtime"`: `trace={}`.
- `execution_mode="debug"`:
  - `detections`: traza por detección (`shirt_crop_available`, `sample_decision`, `relabel`, etc.).
  - `clusters`: snapshot de `TeamColorModel` + eventos de actualización.
  - `summary`: resumen agregado del frame.

## API de fase

`IdentificationPhase.execute(...)` mantiene la API actual:

```python
packet = identification_phase.execute(
    frame_bgr,
    filtering_packet,
    show_kmeans=False,
    execution_mode="runtime",
)
```

`show_kmeans` se conserva por compatibilidad aunque internamente se usa como trigger de visualización de crops/clusters.

## Archivos clave

- `team_detector.py`: orquestación del módulo y construcción del packet `IDENTIFICATION`.
- `team_color_model.py`: bootstrap, actualización de clusters y asignación por color.
- `shirt_detector.py`: extracción de color LAB por crop.
- `team_detector_utils.py`: crops, serialización y gates posicionales.
- `kmeans_batch_gpu.py`: KMeans batch CPU/GPU.
- `phase.py`: wrapper `Phase`.
- `__init__.py`: exports públicos del módulo.

## Nota sobre imports

El paquete usa exports lazy en `__init__.py` para no arrastrar `cv2`, `sklearn`, `scipy` o dependencias GPU al importar `football_ai.tracking.phases.identification` si no hace falta.
