# posession

Fase desacoplada de posesión.

- Entrada: packet `CANONICALTRACK`.
- Salida: packet `POSESSION` con:
  - `clean`: copia de `CANONICALTRACK.clean` + `possession` + `tracks_frame` enriquecido con señales de posesión.
  - `trace`: copia de `CANONICALTRACK.trace` + bloque `possession`.

API pública:

- `PosessionPhase(config_mapping).process_packet(canonical_packet)`
