# visualization

Generación de video anotado con los resultados del tracking.
Soporta dos modos:
- `single`: overlay clásico sobre el frame original.
- `four_panel`: mosaico 2x2 con paneles de depuración por frame.

---

## `drawer.py` — `Drawer`

### Objetivo
Tomar el diccionario `tracks` generado por `Tracker` y producir un nuevo video MP4 con los bounding boxes y etiquetas pintados sobre cada frame original.

### Inicialización

```python
from football_ai.visualization import Drawer

drawer = Drawer(
    colors={
        "player":     (0, 255, 0),    # Verde (BGR para OpenCV)
        "goalkeeper": (0, 255, 255),  # Amarillo
        "referee":    (255, 0, 0),    # Azul
        "ball":       (0, 0, 255)     # Rojo
    },
    default_color=(255, 255, 255)     # Blanco para clases desconocidas
)
```

Los colores se leen de `config.yaml` vía `config.get_visualization_colors()` en los scripts.
En modo campo (`four_panel` panel B), el color de relleno por equipo usa los `team_colors` activos del `TeamDetector` (LAB OpenCV convertidos a BGR al pintar).

### Método principal: `draw_tracks`

```python
drawer.draw_tracks(
    tracks=tracks,          # dict de Tracker.get_tracks()
    video="partido.mp4",    # video original para leer los frames
    output_path="output/resultado.mp4",
    show=False,             # si True, muestra en ventana en tiempo real
    four_panel=False,       # si True, genera mosaico 2x2
    debug_frames=None,      # metadata cruda/descartada (solo four_panel)
    expected_counts=None    # límites por clase para panel de continuidad
)
```

En Linux sin entorno gráfico (sin `DISPLAY` ni `WAYLAND_DISPLAY`), si `show=True` se desactiva automáticamente la visualización en ventana para evitar errores de Qt/xcb; el video de salida se sigue escribiendo.

**Flujo interno:**
1. `create_writer(video, output_path)`: abre el video con `cv2.VideoCapture`, extrae FPS, ancho y alto, y crea un `cv2.VideoWriter` con codec `mp4v`. Crea el directorio de salida si no existe.
2. Itera frame a frame con `cap.read()`.
3. Para cada clase y frame, llama a `draw_all_detections_in_frame`, que itera sobre todos los tracks del frame.
4. `draw_detection` pinta el bounding box con `cv2.rectangle` y la etiqueta con `cv2.putText`. En modo clásico incluye:
   - Clase y track_id (`"player #7"`)
   - Si hay información de equipo: el equipo asignado y las distancias a cada equipo dinámicamente (`"Real Madrid: [12.3, 45.6]"`)
   - Si el tracking viene de un `lineup_spec.json` y el slot ya se ha estabilizado: el `player_name` resuelto y su `lineup_slot`
   - Si hay `predicted_role_frame` o `predicted_role`: una línea adicional bajo el bbox con el rol
   - Si existe `display_role_slot`, prioriza mostrar esa salida final del segundo Hungarian de segmento; si no, cae a `expected_role_slot`, luego a `segment_majority_expected_role_slot` y por último a `segment_majority_role`
   - Si hay `field_position_m` en `player`: otra línea bajo el bbox con `pos(m): x, y`
   - Si el track coincide con el jugador en posesión: dibuja un segundo recuadro amarillo alrededor del bbox
   - Además pinta un banner `POS: <equipo>` en el frame
5. Escribe el frame anotado con `out.write(frame)`.
6. En el bloque `finally`, libera `cap` y `out` siempre, incluso si hubo error.

### Modo `four_panel`

Cuando `four_panel=True`, cada frame de salida se divide en 4 paneles:
1. `A) Tracking compact`: video anotado en formato compacto (`p`, `gk`, `ref`, sin distancias de equipo, roles sin prefijo, posición `x, y` sin decimales y texto más pequeño).
   - Bajo cada track compacto se muestran `tr:<cls>` (clase actual del track), `y:<cls>` (YOLO) y `td:<cls>` (TeamDetector).
   - En `player/gk`, el color de `bbox` se toma de los `team_colors` activos del `TeamDetector` para el equipo resuelto. Si `team` viene vacío, primero intenta resolverlo por `distances`.
2. `B) Campo + IDs + rol`: representación 2D del campo con:
   - círculo por track final (relleno por equipo),
   - borde por tipo (`player`, `goalkeeper`, `referee`),
   - `track_id` dentro y rol debajo,
   - balón como cuadrado rojo,
   - anillo amarillo para el jugador en posesión,
   - banner `POS: <equipo>`.
   - El tamaño de círculo se controla con `visualization.pitch_marker_radius` (por defecto 12).
3. `C) YOLO descartadas`: detecciones crudas de YOLO que no acabaron en un track canónico en ese frame.
   - Se separan en dos tipos (colores configurables en `config.yaml/visualization`):
     - `discarded_panel_color_not_tracked`: YOLO no devueltas por ByteTrack.
     - `discarded_panel_color_tracked_no_canonical`: devueltas por ByteTrack pero descartadas en el mapeo a ID canónico.
   - En `player/gk`, el borde y la etiqueta usan el mismo criterio (equipo resuelto contra `team_colors` activos del `TeamDetector`). Los colores configurables siguen actuando como fallback para `referee/ball`.
   - Ambas etiquetas muestran `tr:<cls>`, `y:<cls>` y `td:<cls>` (track/YOLO/TeamDetector), además de una abreviatura compacta y la confianza.
   - En las detecciones que nunca llegaron a salir de ByteTrack, `tr:-` indica explícitamente que no hubo clase de track disponible.
   - Para las detecciones devueltas por ByteTrack pero descartadas en canónico, el panel incluye también `bt#<id>` (el `tracker_id` devuelto por ByteTrack).
   - Si `visualization.discarded_panel_show_reasons=true`, el panel añade una abreviatura compacta del `discard_reason` en la etiqueta (útil para depurar gates/límites sin desbordar el overlay). El JSON `*_debug_frames.json` sigue guardando el motivo completo.
   - Cuando `four_panel_enabled=true`, el pipeline guarda además un JSON `*_debug_frames.json` junto al `*_tracks.json` con estas listas y motivos, para análisis offline.
4. `D) Tracking con continuidad`: overlay compacto con relleno de continuidad (usa la última posición conocida por ID cuando falta detección en el frame) y mantiene el resaltado de posesión.
   - Igual que el panel A, muestra `tr:<cls>`, `y:<cls>` y `td:<cls>` de cada track y colorea `player/gk` con el color de equipo resuelto en `team_colors` del `TeamDetector`.
   - Añade además `seg:<id>` bajo cada track para mostrar el segmento semántico activo del `canonical_id`.
   - Si `visualization.continuity_keep_all_seen_ids=true`, mantiene visibles todos los IDs ya observados en el clip para cada clase (no corta al cupo esperado).

Compatibilidad de clases en `tracks`:
- El drawer dibuja con clases canónicas (`player`, `goalkeeper`, `referee`, `ball`).
- También normaliza aliases legacy (`ref`, `refs`, `referees`, `gk`, `players`, `balls`) para evitar perder renderizado por naming histórico.

### Manejo de errores

`create_writer` levanta excepciones tipadas:
- `ValueError`: ruta de video vacía.
- `FileNotFoundError`: el archivo de video no existe.
- `RuntimeError`: no se puede abrir el video o crear el writer (codec no disponible, permisos...).

### Métodos auxiliares

| Método | Descripción |
|---|---|
| `create_writer(video, output_path)` | Prepara `VideoCapture` y `VideoWriter`, añade `.mp4` si falta extensión |
| `draw_detection(frame, class_name, data, color, track_id)` | Dibuja un único bbox con etiqueta |
| `draw_all_detections_in_frame(frame, class_name, class_tracks, frame_id)` | Dibuja todos los tracks de una clase en un frame |
| `draw_tracks(tracks, video, output_path, show, window_name, four_panel, debug_frames, expected_counts)` | Pipeline completo (clásico o 2x2) |

> **Nota:** El método `draw_detection` itera dinámicamente sobre las claves del diccionario `distances` para formatear las distancias, por lo que es compatible con cualquier configuración de equipos en config.yaml.

---

## `pathcrf_drawer.py` — `PathCRFDrawer`

### Objetivo

Renderizar PathCRF a partir del parquet ancho y de la secuencia de aristas inferida (`edge_src`, `edge_dst`). Si el pipeline conoce también el vídeo original y el `tracks.json`, el drawer pinta sobre el broadcast real usando las `bbox` reales e incrusta un mini-mapa 2D en la esquina superior derecha. Si no dispone de eso, cae al modo 2D puro.

### Método principal

```python
from football_ai.visualization import PathCRFDrawer

drawer = PathCRFDrawer()
drawer.render_tracking_and_edges(
    tracking=tracking_df,
    edge_sequence=edge_seq_df,
    output_path="output/actions/pathcrf/partido/partido_pitch_pathcrf.mp4",
    events=events_df,
    fps=25.0,
    frame_size=(1280, 720),
    show=False,
)
```

### Qué pinta

- sobre broadcast real, las `bbox` reales de `player`, `goalkeeper`, `referee` y `ball` cuando existen;
- resaltado visual de `edge_src` y `edge_dst` sobre esas cajas si se puede mapear el slot PathCRF al `raw_track_id`;
- mini-mapa 2D semitransparente en la esquina superior derecha con:
  - nodos `home_1..11` y `away_1..11` coloreados por equipo;
  - árbitros `referee_1..3` como nodos grises;
  - nodos exteriores de PathCRF (`out_left`, `out_right`, `out_bottom`, `out_top`);
  - balón estimado (`ball_x`, `ball_y`);
- arista activa del frame:
  - flecha si `edge_src != edge_dst`;
  - anillo de control si es self-loop;
- overlay con `frame`, `timestamp`, arista activa y, si toca exactamente en ese frame, el evento derivado (`kick`, `control`, `out`, ...).

### Uso típico

No suele llamarse a mano: `scripts/actions/run_pathcrf.py` lo usa automáticamente al final del pipeline salvo que se pase `--no-render`.
