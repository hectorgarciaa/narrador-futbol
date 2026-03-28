# possession

Experimentos para inferir posesion de balon por equipo a partir del tracking ya generado.

## `team_possession.py`

**Objetivo:** estimar por frame que equipo tiene la posesion y renderizar una cartela fija en una esquina del video con ese equipo.

Heuristica implementada:
- toma el centro del bbox del balon por frame;
- filtra el track del balon con una trayectoria esperada basada en posiciones anteriores y velocidad estimada;
- cuando una deteccion del balon pega un salto implausible, la rechaza y usa una prediccion corta de la trayectoria en vez de aceptar ese outlier;
- busca el `player/goalkeeper` mas cercano al balon usando el punto inferior central del bbox;
- confirma toque cuando hay proximidad estricta y alguna pista de control:
  - velocidad baja del balon,
  - caida de velocidad,
  - cambio de direccion,
  - balon dentro del bbox expandido,
  - o robo claro del rival;
- exige confirmacion temporal para muchos cambios de equipo rivales, para evitar saltos de un solo frame;
- rellena lagunas cortas y suprime segmentos muy breves de posesion para estabilizar la salida visual;
- mantiene siempre la posesion del ultimo equipo que toco el balon, para cubrir pases en transito y balones fuera de vision;
- solo cambia de equipo cuando hay contacto plausible y senal en la velocidad y/o direccion del balon.

Salidas:
- `output/predictions/possession/<video>_<timestamp>/frame_possession.csv`
- `output/predictions/possession/<video>_<timestamp>/frame_possession.json`
- `output/predictions/possession/<video>_<timestamp>/summary.json`
- `output/predictions/possession/<video>_<timestamp>/<video>_possession_annotated.mp4`

Ejecucion:

```bash
python -m experiments.possession.team_possession \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```
