# tracking

Pipeline de tracking principal del proyecto.

## Que hace

Orquesta las fases que convierten un frame de video en tracks canonicos con IDs estables, equipo, posicion proyectada y metadatos utiles para modulos posteriores.

## Flujo

`DETECTION -> REFERENCE_POINTS -> FILTERING -> IDENTIFICATION -> BYTETRACK -> CANONICALTRACK`

Despues de ese bloque, el proyecto puede anadir posesion, roles y comentarios segun la configuracion.

## Archivos clave

- `tracker.py`: orquestador del pipeline por frame.
- `phase.py`: wrapper `TrackingPhase`.
- `phases/`: implementaciones de las fases desacopladas.

## Salida

La salida principal del modulo es `tracks_frame`, que luego `scripts/track.py` acumula y exporta a JSON y video anotado.
