# interfaz

Interfaz web ligera para preparar alineaciones y lanzar el tracking del partido.

## Qué hace

- permite introducir dos equipos
- permite definir color de camiseta por equipo
- permite elegir formación (`4-3-3`, `5-3-2`, `4-4-2`)
- renderiza un campo más detallado con marcas reglamentarias y slots clicables
- guarda un `lineup_spec.json` por ejecución
- lanza `scripts/track.py --lineup-spec ...`
- arranca automáticamente el servidor de comentarios al abrir la interfaz
- por defecto intenta usar `llama.cpp` leyendo `external/llama.cpp/config.yaml` y lanza `llama-server` automáticamente antes de precalentar el `intro`
- el precalentado del `intro` y del servidor de comentarios corre en segundo plano, así que la interfaz HTTP queda disponible sin esperar a XTTS ni al primer warmup de Gemma
- permite elegir modo de comentarios `live` o `deferred` (por defecto `live`)
- precalienta un comentario de `intro` al arrancar la interfaz para que ya esté listo al guardar la alineación
- si el `intro` sale con una plantilla absurda o el servidor reutilizado es antiguo, la interfaz lo regenera localmente antes de guardarlo
- obliga a completar primero los dos nombres de equipo antes de desbloquear vídeo, modo de comentarios, colores, formación y jugadores
- no recrea las tarjetas mientras escribes el segundo nombre, para que el formulario no se desbloquee ni te robe el foco a mitad de la edición
- muestra el vídeo final dentro de la propia interfaz cuando el MP4 ya está listo
- al resolver los vídeos de resultado (`tracking`, `PathCRF`, `commentary`), solo enseña archivos generados o actualizados por la propia run; si una ejecución no regenera `*_pathcrf.mp4`, la interfaz no reutiliza un PathCRF viejo de otra ejecución anterior del mismo vídeo
- si la ejecución nace desde la interfaz, activa un bridge incremental `tracking -> PathCRF -> servidor de comentarios` en segundo plano, sin afectar a `scripts/track.py` cuando se ejecuta suelto
- muestra el estado del proceso y el log en vivo

## Cómo se ejecuta

Desde la raíz del proyecto:

```bash
python interfaz/app.py
```

Opciones:

```bash
python interfaz/app.py --host 127.0.0.1 --port 8767
python interfaz/app.py --commentary-backend llama_cpp --commentary-port 8788
python interfaz/app.py --commentary-backend ollama --commentary-model gemma4:e2b
```

Después abre la URL que imprime la consola al arrancar. Por defecto será `http://127.0.0.1:8767`, pero si ese puerto ya está ocupado la interfaz probará el siguiente puerto libre y mostrará el definitivo.

Nota:

- la interfaz expone también respuestas `HEAD` para mejorar compatibilidad con navegadores como Safari;
- si trabajas en una máquina remota, `127.0.0.1` debe estar reenviado a tu equipo local o abrirse con la IP/host remotos.
- `--commentary-backend auto` intenta usar `llama.cpp` si existe `external/llama.cpp/config.yaml`; si no, cae a `ollama`.
- el autoarranque del backend LLM solo se intenta cuando `--commentary-base-url` apunta a una URL local; si apuntas a un backend remoto, la interfaz solo lo reutiliza.
- la configuración de `llama.cpp` vive en `external/llama.cpp/config.yaml`; ahí se fija el binario `llama-server`, el alias, el puerto y el GGUF a cargar.

## Qué genera

Cada ejecución crea:

- `output/interfaz/runs/<run_id>/lineup_spec.json`
- `output/interfaz/runs/<run_id>/status.json`
- `output/interfaz/runs/<run_id>/track.log`
- `output/interfaz/runs/<run_id>/commentaries/events_manifest.jsonl`
- `output/interfaz/runs/<run_id>/commentaries/audio/000_intro.wav`
- `output/interfaz/runs/<run_id>/commentaries/commentary_track.wav` cuando el vídeo termina y se ensambla la pista diferida

Cuando el tracking se lanza desde la interfaz, la fase `CommentaryPhase` del subprocess escribe directamente en ese mismo `output/interfaz/runs/<run_id>/commentaries/`. Ya no reutiliza `output/actions/rolling_online/<video>/commentary/`, así que el manifiesto y los WAV del run quedan aislados y el MP4 final no arrastra audios de partidos anteriores.

El `<run_id>` incluye fecha/hora local, equipos, vídeo y un sufijo corto para que se pueda reconocer la salida al mirar la carpeta. Ejemplo: `20260413-153012_madrid-vs-barcelona_video-prueba-ajustado_a1b2`.

El propio `track.py` copia además el spec dentro del directorio de artefactos del vídeo para dejar trazabilidad completa.

## Modo comentarios

- `live`: la interfaz intenta reproducir el `intro` ya precalentado nada más guardar y luego hace polling del manifiesto para sonar el audio más reciente disponible cuando el canal esté libre, sin cola FIFO de audios antiguos.
- `deferred`: se siguen generando y guardando comentarios/audio, pero la interfaz no los reproduce al vuelo. Al terminar el tracking, la interfaz construye una pista continua desde el manifiesto y la incrusta en el MP4 final del tracking.
- la intro y los comentarios del partido respetan `commentary.alternate_voices` de `config.yaml`: si esta desactivado, se usa solo la voz principal; si esta activado y existe voz femenina, se alternan ambas con aleatoriedad controlada.
- el bridge PathCRF evita reenvíos casi idénticos, guarda comentarios de texto aunque el turno de audio esté ocupado y solo solicita un nuevo WAV si el `event_time_s` cae después de la ventana ocupada por la generación más la duración del audio anterior.
- PathCRF etiqueta cada acción con zona de campo (`iniciacion`, `creacion`, `finalizacion`) según el tercio del largo y la dirección de ataque; las acciones en iniciación pueden generar de forma puntual un comentario de contexto con datos tácticos del `lineup_spec.json` y clasificación simulada.
- si un comentario de contexto está sonando y aparece una acción urgente (`tiro` o `gol`), el nuevo audio se marca como interrupción y el navegador corta el contexto para reproducir la acción peligrosa.
- al ensamblar el vídeo diferido, los WAV del manifiesto se colocan en su `event_time_s` real y los que se solaparían con un audio ya aceptado se omiten de la pista sonora, manteniendo el comentario escrito; un tiro/gol puede sustituir a un contexto interruptible si se pisan.
- el servidor de comentarios omite duplicados consecutivos cuando llega otra vez la misma `action` para el mismo `player_name` y equipo.
- el MP4 final diferido se vuelve a codificar como `H.264/AAC` con perfil compatible y límite `1080p`, así que el archivo que sirve la interfaz es reproducible por navegadores modernos y no se queda en negro por usar `mp4v` o por exportar un H.264 4K demasiado pesado.
- si lanzas un run demasiado pronto y el `intro` todavía sigue en `starting`, el tracking arranca igualmente; simplemente ese run puede salir sin el `intro` precopiado.

Cuando el run se lanza desde la interfaz, el propio subprocess de tracking recibe por entorno la información necesaria para:

- ejecutar PathCRF de forma incremental sobre snapshots acumulados del `tracks` mientras avanza el partido;
- transformar los eventos semánticos nuevos en payloads de comentario;
- aplicar una coherencia básica de posesión antes de narrar: pases a rivales pasan a `robo` del receptor y acciones de jugadores que ya no deberían tener el balón se saltan;
- usar la zona de campo para priorizar peligro y alternar narración de acción con contexto de partido en zonas de bajo riesgo;
- enviarlos al mismo servidor HTTP de comentarios que ya usa la interfaz;
- dejar esos textos y los audios aceptados en el manifiesto para `live` o para el ensamblado final en `deferred`.

Para depurar el servidor de comentarios lanzado junto a la interfaz:

```bash
curl http://127.0.0.1:8767/api/commentary-service
```
