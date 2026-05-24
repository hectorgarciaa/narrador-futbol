# bytetrack

Fase de asociacion temporal basada en ByteTrack.

## Que hace

- recibe detecciones ya enriquecidas por `IDENTIFICATION`;
- enlaza detecciones entre frames;
- aplica gates de clase, equipo, tamano y posicion;
- devuelve tracks temporales para la capa canonica.

## Archivos clave

- `byte_tracker.py`: estado principal del tracker.
- `association_costs.py`: costes y mascaras de matching.
- `pipeline_steps.py`: pasos de asociacion del frame.
- `new_track_filter.py`: filtro de nuevos tracks.
- `phase.py`: integracion como fase `BYTETRACK`.

