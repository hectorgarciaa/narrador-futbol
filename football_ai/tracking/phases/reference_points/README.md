# reference_points

Módulo de proyección de detecciones de imagen a coordenadas métricas de campo.

En el pipeline de tracking, esta fase consume el packet `DETECTOR` y emite el packet `REFERENCE_POINTS`.

## Objetivo

1. Estimar una homografía imagen->campo con PnLCalib para el frame actual.
2. Convertir cada detección a una posición de campo (`field_positions_m`) usando un ground point por bbox.
3. Publicar señales de validez/uso para que fases posteriores (`filtering`, `identification`, `bytetrack`, `canonicaltrack`) decidan si usar o no geometría.

## Entrada esperada

`detector_packet["clean"]` (contrato estable del módulo `detection`):

```python
{
    "num_detections": int,
    "det_id": list[int],
    "bbox_xyxy": list[list[float]],
    "confidence": list[float],
    "class_name": list[str],  # solo: player, goalkeeper, referee, ball
}
```

## Salida (`REFERENCE_POINTS`)

Campos en `clean`:

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

Semántica de flags:

- `homography_valid`: existe una homografía publicable para el frame.
- `field_positions_usable_for_tracking`: la homografía además pasó los gates de calidad para uso de tracking.

Ambos flags se mantienen separados por diseño.

## Flujo principal

1. `ProjectionPhase.execute(...)` (en `phase.py`) delega en `PnLCalibFieldProjector.project_frame(...)` si hay proyector activo.
2. El frame se redimensiona (ancho máximo configurable) para inferencia PnLCalib.
3. Se calcula `ground_points_image_original` por detección:
   - `ball` usa offset vertical `0.0`.
   - clases de persona (`player`, `goalkeeper`, `referee`) usan `bottom_offset_ratio` configurable.
4. Se ejecuta un forward de PnLCalib y se evalúan intentos adaptativos de thresholds (`keypoint_threshold`, `line_threshold`).
5. Se selecciona el mejor intento por aceptación y score, con opcional smoothing temporal.
6. Se proyectan los ground points al campo (`field_positions_m`) si hay homografía válida; si no, se devuelven ceros.
7. En `debug` se adjuntan trazas completas (`attempts`, `diagnostics`, `keypoints`, `lines`).

## Fallback sin homografía

Hay dos casos de fallback:

1. `ProjectionPhase({"enabled": False})`: modo sin PnLCalib (pipeline sigue emitiendo packet compatible).
2. Error recuperable durante proyección de un frame: se devuelve packet sin homografía para no cortar el vídeo.

El helper `build_reference_points_packet_without_homography(...)`:

- mantiene el mismo esquema de `clean`,
- usa homografía identidad en `homography_image_to_field_3x3`,
- marca `homography_valid=False` y `field_positions_usable_for_tracking=False`,
- calcula `ground_points_image_original` con el `bottom_offset_ratio` recibido (default `0.04`).

## Calidad de proyección (`quality.py`)

`ProjectionQualityAnalyzer` calcula un score compuesto con tres bloques:

1. `geometry_fit`: reprojection error, conditioning de homografía y alineación mundo.
2. `support_quality`: cantidad/confianza/diversidad de keypoints y líneas visibles.
3. `coverage_quality`: cobertura espacial en imagen y cobertura semántica en campo.

Un intento se acepta (`quality_status="good"`) solo si supera thresholds mínimos y score global.

## Runtime vs debug

- `execution_mode="runtime"`: salida mínima en `trace` (`{}`).
- `execution_mode="debug"`: se incluyen detalles diagnósticos extensos.

Si ocurre un error recuperable en debug, `trace["diagnostics"]` incluye:

- `error_type`
- `error_message`
- `rejection_type="runtime_error"`
- `quality_status="runtime_error"`

## Carga de PnLCalib (`runtime_loader.py`)

`load_pnlcalib_runtime(...)`:

1. valida dependencias requeridas,
2. asegura repo externo `external/pnlcalib` (clonado si falta),
3. asegura pesos en `models/pnlcalib`,
4. aplica workarounds de import (`sys.path`, limpieza de módulos conflictivos, registro manual de paquetes),
5. carga modelos keypoints/líneas y devuelve un `PnLCalibRuntime`.

`project_root` se usa para resolver rutas de repo y pesos de forma estable.

## Geometría y sistema métrico

- `PitchGeometry` define el sistema canónico del pipeline (`106 x 68 m`).
- El template interno de PnLCalib es `105 x 68 m`; la conversión/escala se maneja explícitamente en el módulo.
- `points_inside_field_mask(...)` conserva `geometry=None` como conveniencia para utilidades/tests, pero el flujo principal pasa `geometry` explícitamente.

## Archivos clave

- `projector.py`: proyector principal y construcción del packet.
- `phase.py`: wrapper de fase para el pipeline.
- `estimation.py`: forward e inferencia PnLCalib por frame.
- `quality.py`: validación y scoring de homografía.
- `geometry.py`: utilidades de homografía/proyección.
- `runtime_loader.py`: bootstrap y carga robusta del runtime PnLCalib.
- `common.py`, `pitch_layout.py`: geometría del campo y referencias semánticas.
