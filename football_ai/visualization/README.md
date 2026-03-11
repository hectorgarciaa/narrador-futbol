# visualization

Generación de video anotado con los resultados del tracking: bounding boxes por clase, etiquetas con el ID del track, equipo asignado y distancias a los colores de referencia.

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

### Método principal: `draw_tracks`

```python
drawer.draw_tracks(
    tracks=tracks,          # dict de Tracker.get_tracks()
    video="partido.mp4",    # video original para leer los frames
    output_path="output/resultado.mp4",
    show=False              # si True, muestra en ventana en tiempo real
)
```

En Linux sin entorno gráfico (sin `DISPLAY` ni `WAYLAND_DISPLAY`), si `show=True` se desactiva automáticamente la visualización en ventana para evitar errores de Qt/xcb; el video de salida se sigue escribiendo.

**Flujo interno:**
1. `create_writer(video, output_path)`: abre el video con `cv2.VideoCapture`, extrae FPS, ancho y alto, y crea un `cv2.VideoWriter` con codec `mp4v`. Crea el directorio de salida si no existe.
2. Itera frame a frame con `cap.read()`.
3. Para cada clase y frame, llama a `draw_all_detections_in_frame`, que itera sobre todos los tracks del frame.
4. `draw_detection` pinta el bounding box con `cv2.rectangle` y la etiqueta con `cv2.putText`. La etiqueta incluye:
   - Clase y track_id (`"player #7"`)
   - Si hay información de equipo: el equipo asignado y las distancias a cada equipo dinámicamente (`"Real Madrid: [12.3, 45.6]"`)
   - Si hay `field_position_m` en `player`: una segunda línea bajo el bbox con `pos(m): x, y`
5. Escribe el frame anotado con `out.write(frame)`.
6. En el bloque `finally`, libera `cap` y `out` siempre, incluso si hubo error.

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
| `draw_tracks(tracks, video, output_path, show, window_name)` | Pipeline completo |

> **Nota:** El método `draw_detection` itera dinámicamente sobre las claves del diccionario `distances` para formatear las distancias, por lo que es compatible con cualquier configuración de equipos en config.yaml.
