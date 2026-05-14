# filtering

Tercera fase del pipeline de tracking.  
Consume `REFERENCE_POINTS` y emite `FILTERING`.

## Objetivo

Filtrar detecciones proyectadas para que las fases posteriores trabajen con observaciones geométricamente coherentes, manteniendo compatibilidad de packet y sin romper continuidad cuando la homografía no es usable.

## Entrada esperada

`reference_packet["clean"]`:

```python
{
    "num_detections": int,
    "det_id": list[int],
    "bbox_xyxy": list[list[float]],
    "confidence": list[float],
    "class_name": list[str],
    "homography_valid": bool,
    "homography_image_to_field_3x3": list[list[float]],
    "field_positions_m": list[list[float]],
    "ground_points_image_original": list[list[float]],
    "field_positions_usable_for_tracking": bool,
}
```

Las clases están normalizadas (`player`, `goalkeeper`, `referee`, `ball`).

## Salida (`FILTERING`)

`clean` mantiene solo las detecciones aceptadas:

```python
{
    "num_detections": int,
    "det_id": list[int],
    "bbox_xyxy": list[list[float]],
    "confidence": list[float],
    "class_name": list[str],
    "field_positions_m": list[list[float]],
    "ground_points_image_original": list[list[float]],
}
```

## Regla de filtrado

Fuente de verdad geométrica: `field_positions_m` (no se reproyecta localmente).

1. Si `homography_valid == False` o `field_positions_usable_for_tracking == False`:
   - no se filtra por geometría,
   - se conservan todas las detecciones.
2. Si la homografía es usable:
   - `geometry` es obligatoria (`ValueError` si falta),
   - una detección se mantiene si:
     - su posición es finita, y
     - cae dentro del campo o dentro de la banda lateral permitida (`sideline_margin_m`).
3. Si está fuera:
   - puede rescatarse por solape con tracks activos (`IoU > rescue_iou_threshold`).

## Parámetros principales

- `sideline_margin_m` (default `0.75`): margen vertical alrededor de laterales para no penalizar pequeñas desviaciones de proyección.
- `rescue_iou_threshold` (default `0.0`): umbral para rescate por IoU con cajas activas.
- `active_track_boxes_xyxy`: cajas activas para rescate.
- `geometry`: requerida solo cuando la homografía es usable.
- `execution_mode`: `runtime` o `debug`.

## Runtime vs Debug

- `runtime`: `trace={}` (sin coste de construir trazas detalladas).
- `debug`: `trace` incluye:
  - `accepted_detections`
  - `rejected_detections`
  - `summary` (`total_before_filter`, `total_kept`, `total_rejected`, `total_rescued_by_iou`, flags de homografía, `rescue_iou_threshold`).

## Estructura del módulo

- `post_projection.py`
  - `filter_reference_points(...)`: lógica principal.
  - `_tlbr_iou(...)`: IoU entre cajas TLBR para rescate.
- `phase.py`
  - `FilteringPhase`: wrapper `Phase` usado por el orquestador.
- `__init__.py`
  - exporta `filter_reference_points` y `FilteringPhase`.

## Notas de diseño

- Se mantiene el nombre `filter_reference_points` por compatibilidad, aunque conceptualmente filtra detecciones proyectadas.
- Existe acoplamiento con `reference_points.geometry.points_inside_field_mask`; es aceptable hoy, pero mover utilidades geométricas a un módulo común sería una mejora futura.
