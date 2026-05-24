# detection

Fase de deteccion del pipeline de tracking.

## Que hace

- ejecuta YOLO sobre cada frame;
- normaliza clases a `player`, `goalkeeper`, `referee` y `ball`;
- publica un packet simple con `bbox`, confianza y clase.

## Archivos clave

- `detector.py`: wrapper de inferencia.
- `phase.py`: integracion como fase `DETECTION`.
- `utils.py`: helpers de normalizacion.

