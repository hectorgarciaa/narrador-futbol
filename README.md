# 🎙️ AI Football Commentator

Sistema de inteligencia artificial para narración automática de partidos de fútbol, combinando detección de objetos, tracking multi-objeto, identificación de equipos, reconocimiento de acciones, generación de comentarios con LLM y síntesis de voz.

> **Estado actual:** Fase 1 completada (detección, tracking e identificación de equipos). Fases 2-5 pendientes.

---

## 📖 Descripción del Proyecto

El objetivo es construir un **pipeline completo de narración automática de fútbol** capaz de:

1. Detectar jugadores, árbitros y balón en cada frame del vídeo.
2. Hacer tracking multi-objeto para mantener IDs persistentes entre frames.
3. Identificar a qué equipo pertenece cada jugador por el color de camiseta.
4. Reconocer acciones del partido (pases, tiros, goles, faltas…).
5. Generar comentarios expresivos y contextualizados con un LLM.
6. Convertir los comentarios en audio para narración en tiempo real.

---

## 🏟️ Fases del Proyecto

### ✅ Fase 1: Detección, tracking e identificación de equipos
- Fine-tuning de YOLOv11 para las clases `player`, `goalkeeper`, `referee`, `ball`.
- La detección frame a frame sigue ahora un contrato fijo `clean/trace`: `Detector` filtra desde el primer momento a las clases soportadas (`player`, `goalkeeper`, `referee`, `ball`) y remapea aliases comunes a esas clases canónicas.
- Tracking multi-objeto con **ByteTrack** extendido con penalización por equipo, doble señal de clase (YOLO + reetiquetado por color) y remapeo controlado por consenso.
- Proyección automática al campo 2D con **PnLCalib** antes de la identificación de equipos; usa anclajes por clase (`player`/`goalkeeper`/`referee` en pie y `ball` sin offset vertical) y emplea posiciones métricas de `player` y `goalkeeper` en el matching del tracker solo cuando la homografía del frame supera una validación explícita basada en `geometry_fit`, `support_quality` y `coverage_quality`. El proyector puede relajar thresholds para rescatar el frame, compara todos los intentos por score y solo conserva la homografía si queda clasificada como `good`; si no, el pipeline cae a bbox y no usa `field_position_m` para decisiones de tracking/canonización.
- El runtime de `PnLCalib` se resuelve ahora siempre desde `project_root` del proyecto: si falta el repositorio externo se clona en `external/pnlcalib/` y sus pesos se descargan en `models/pnlcalib/` sin depender del directorio actual desde el que se lance el tracker.
- Si existe homografía válida del frame anterior, el proyector puede aplicar `temporal_blend` como suavizado temporal, pero ese blend solo se adopta cuando también supera la validación de calidad y no empeora el `quality_score`; en caso contrario se mantiene la homografía actual sin suavizar.
- El pipeline visual y de tracking encadena seis packets por frame: `DETECTOR`, `REFERENCE_POINTS`, `FILTERING`, `IDENTIFICATION`, `BYTETRACK` y `CANONICALTRACK`. `clean` se usa para la lógica del pipeline y `trace` queda reservado a auditoría/debug. En `tracking.execution_mode=runtime`, las fases ya no construyen trazas ricas.
- `FILTERING` sí elimina detecciones en `clean`: conserva el mismo esquema que `REFERENCE_POINTS`, pero solo con las detecciones aceptadas. El detalle de aceptadas/rechazadas y su `reject_code` queda separado en `trace`.
- Identificación de equipo mediante **KMeans en espacio LAB** sobre el crop de camiseta.
- Inferencia online de roles futbolísticos con Set Transformer. El modelo trabaja con 11 roles tras fusionar carrileros con laterales (`CI -> LI`, `CD -> LD`); en alineaciones y `expected_roles_by_team` deben usarse directamente `LI` y `LD`. La segunda pasada de Húngaro del tracking reaplica el slot final por segmento real `(track_id, segment_id)` para evitar colisiones entre jugadores distintos de un mismo equipo. Los segmentos siguen rompiéndose por señales duras de relink/cambio estructural, mientras que los cambios tácticos temporales de rol se absorben con el sistema de votos del propio segmento.
- La fase de comentarios ya no reutiliza un manifiesto global por vídeo: cada ejecución escribe sus eventos y audios en un directorio propio de run, evitando mezclar nombres o `Jugador N` heredados de ejecuciones anteriores al montar el MP4 final con narración.
- `IDENTIFICATION` consume `FILTERING.clean` y devuelve un `clean` enriquecido con `class_td`, `team`, `shirt_color`, `distances` y `bbox_size`, manteniendo la clase YOLO original en `class_name`. Su `trace` solo se construye en `tracking.execution_mode=debug` e incluye el detalle por detección, motivos de relabel y estado/eventos de clustering.
- `IDENTIFICATION` mantiene la semántica histórica del tracker: usa `field_positions_m` cuando cada detección trae coordenadas finitas, y solo cae a `field_position=None` si la posición no existe o no es finita.
- `BYTETRACK` consume `IDENTIFICATION.clean` como fase independiente y asume el contrato alineado por detección (`det_id`, `bbox_xyxy`, `confidence`, `class_name`, `class_td`, `team`, `distances`, `shirt_color`, `bbox_size`, `field_positions_m`, `ground_points_image_original`) sin añadir validación defensiva extra en runtime. Devuelve un `clean` mínimo para la canonización (`det_id`, `bbox_xyxy`, `confidence`, `class_name`, `field_positions_m`, `ground_points_image_original`, `tracked_detections`). La alineación detección↔track y las trazas ricas de matching viven en `trace`, y solo aparecen en `debug`.
- `CANONICALTRACK` consume exclusivamente `BYTETRACK.clean`, aplica canonización/relink/absorción forzada/seeds/selección de balón y devuelve `tracks_frame` por clase. Sus trazas de descarte y diagnóstico (`pending_assignments_debug`, `discard_reason_by_raw_idx`, `forced_absorption_debug`, `ball_selection_debug`) solo se construyen en `tracking.execution_mode=debug`.
- Gate posicional para el relabel `player -> referee`: una detección solo puede convertirse en árbitro por color si, tras la homografía, cae en la franja lateral válida o entre la cuarta `x` más a la izquierda y la cuarta más a la derecha de los jugadores visibles.
- Anti-solape de ByteTrack limitado al nacimiento de tracks nuevos: los `unconfirmed` ya nacidos siguen el matching normal y el filtro duro de solape solo se aplica antes de crear un track nuevo frente a activos, `unconfirmed` previos y otros candidatos del mismo frame, con thresholds independientes para cada comparación.
- Sistema de evaluación cuantitativo por track (cobertura, fragmentación, velocidad, etc.).

### 🚧 Fase 2: Detección de acciones
- Seleccionar y adaptar una red preentrenada para detectar acciones de fútbol (pase, tiro, gol, falta, tarjeta, penal…).
- El flujo rolling de acciones se ejecuta ahora en modo snapshot (adapter offline por cadencia) con `scripts/actions/run_rolling_pathcrf.py`; `--tracking-path` queda solo como modo diagnóstico.

### 📅 Fase 3: Generación de comentarios con LLM
- Integrar información de tracking y acciones y enviarla a un LLM para generar comentarios expresivos y contextualizados.

### 📅 Fase 4: Conversión de texto a audio
- Transformar los comentarios a audio con una solución de TTS.

### 📅 Fase 5: Pipeline en tiempo real
- Integrar todas las fases en un pipeline eficiente para retransmisión en vivo.

---

## 🗂️ Estructura del Proyecto

```
narrador-futbol/
├── config.yaml             # Configuración central (rutas, hiperparámetros, equipos)
├── pyproject.toml          # Metadatos del paquete Python
├── requirements.txt        # Dependencias
├── clean_project.py        # Script de limpieza de archivos generados
├── verify_setup.py         # Script para verificar que todo esta listo
│
├── football_ai/            # Paquete principal (toda la lógica de negocio)
│   ├── core/               # Configuración, logging, serialización
│   ├── actions/            # Adaptador + inferencia PathCRF offline y rolling live
│   ├── detection/          # Wrapper YOLO + cabeza DetectR8 para balón
│   ├── positions/          # Lógica de roles posicionales y estabilización online
│   ├── bytetrack/         # Fase ByteTrack desacoplada (packet IDENTIFICATION -> BYTETRACK)
│   ├── canonicaltrack/    # Fase canónica desacoplada (packet BYTETRACK -> CANONICALTRACK)
│   ├── tracking/          # Capa canónica/orquestador posterior a ByteTrack
│   ├── identification/     # ShirtDetector (KMeans LAB) + TeamDetector
│   ├── evaluation/         # Métricas por track y comparador de experimentos
│   └── visualization/      # Drawer: genera vídeo anotado
│
├── scripts/                # Scripts ejecutables de línea de comandos
│   ├── detect.py           # Detección YOLO simple por CLI
│   ├── homography.py       # Detección + PnLCalib + JSON/vídeo a pantalla partida
│   ├── comparar_modelos.py # Comparativa de checkpoints YOLO en espacio canónico común
│   ├── track.py            # Pipeline completo: tracking + evaluación + vídeo
│   ├── track_experiments.py# Grid search de hiperparámetros del tracker
│   ├── data/
│   │   ├── download_models.py    # Descarga modelos YOLO base
│   │   └── download_datasets.py  # Descarga datasets desde Roboflow
│   └── train/
│       └── finetune_player.py    # Fine-tuning de YOLO para fútbol
│
├── data/                   # Datos de entrada (ver data/README.md)
│   ├── detection/          # Datasets de detección (formato YOLOv11)
│   ├── partidoPrueba/      # Vídeos de partido para tracking/detección
│   └── partidosPosiciones/ # Clips para construir dataset de roles posicionales
│
├── models/                 # Pesos y checkpoints del proyecto (ver models/README.md)
│   ├── yolo/               # Modelos base YOLOv8 y YOLOv11
│   ├── finetuning/         # Modelo fine-tuned de jugadores
│   └── finetuning-balon/   # Modelo fine-tuned de balón
│
├── interfaz/               # UI web ligera para introducir alineaciones y lanzar tracking
│   ├── app.py              # Backend HTTP sin dependencias extra
│   └── static/             # HTML/CSS/JS del editor de alineaciones
│
├── football_ai/commentaries/ # Generacion local de comentarios sinteticos con Ollama
│
└── experiments/            # Notebooks de análisis y visualización
    ├── detection/
    ├── set_transformer.ipynb # Entrenamiento + aplicación de Set Transformer para roles
    ├── positions/          # Dataset supervisado de roles por posición (experimental)
    ├── reference_points/
    ├── tracking/
    └── visualization/
```

  Nota de repositorio: [output/](output/) se ignora por defecto para no subir artefactos pesados. Solo se versionan [output/detection/](output/detection/), [output/homography/](output/homography/) y [output/tracking/](output/tracking/) (ver [.gitignore](.gitignore)).

Nota de arquitectura: el runtime productivo vive en `football_ai/` y `experiments/` actúa como capa de experimentación sobre ese runtime (sin dependencias inversas desde `football_ai` hacia `experiments`).
La configuración operativa del pipeline visual está separada por fases en bloques top-level como `projector`, `bytetracker`, `canonical`, `positions`, `actions`, `commentary` y `posession`; el orquestador compone en runtime la configuración concreta que necesita cada fase.

Los notebooks de `experiments/visualization/` que inspeccionan el relabel de equipos deben apoyarse en las fases actuales del tracker (`detection_phase`, `projection_phase`, `filtering_phase`, `identification_phase`) y no en atributos legacy del `Tracker`. En concreto, `team_detector_relabel_flow_utils.py` ya está adaptado a ese flujo y el notebook `team_detector_relabel_flow_ucl.ipynb` usa el clip UCL correcto también en las celdas de overlay final.

---

## 🧪 Depuración de tracking (four-panel)

El vídeo de salida se genera siempre como mosaico 2x2. Cuando `tracking.execution_mode=debug`, el pipeline guarda además un JSON con metadatos por frame para depurar por qué se pierden IDs:

- `output/tracks_json/tracker/<video>_debug_frames.json`

Incluye, por frame:
- detecciones crudas YOLO,
- detecciones YOLO descartadas porque ByteTrack no las devolvió,
- detecciones devueltas por ByteTrack pero descartadas en el mapeo a IDs canónicos (incluye `bt#<id>` y opcionalmente un `discard_reason`).

En la salida 2x2, los paneles compacto/continuidad muestran además `tr:<cls>`, `y:<cls>` y `td:<cls>` para distinguir la clase actual del track, la clase YOLO y la clase relabelada por `TeamDetector`. El panel inferior izquierdo usa ese mismo trío de etiquetas en las detecciones descartadas junto a la confianza; cuando una detección no llegó a salir de ByteTrack, aparece como `tr:-`. Si activas `visualization.discarded_panel_show_reasons`, el vídeo muestra una versión compacta del `discard_reason` para que quepa en overlay, mientras que el JSON `*_debug_frames.json` conserva el motivo completo.
Además, los paneles superior izquierdo, superior derecho e inferior derecho colorean `player/gk` usando los `team_colors` activos del `TeamDetector` (centros/refs LAB del clustering convertidos a BGR para render). Si un track no tiene equipo resoluble, caen al color por clase.
Ese JSON también puede incluir `frame_num`, `bytetrack_reason` y `bytetrack_stage` para auditar por qué una detección no llegó a consolidarse. La información de homografía rica vive en `REFERENCE_POINTS.trace` solo en `debug`; en runtime, el pipeline conserva en `clean` únicamente las señales funcionales que necesita el resto de fases.

Para un resumen offline rápido puedes usar:

```bash
python scripts/analyze_debug_frames.py output/tracks_json/tracker/<video>_debug_frames.json
```

Para auditorías específicas de homografía frame a frame sobre vídeos concretos, existe además:

```bash
.venv/bin/python football_ai/pruebas/analyze_homography_videos.py \
  football_ai/.data/partidoPrueba/ucl_final_2-7.mp4 \
  football_ai/.data/partidoPrueba/clasico_5-10.mp4 \
  --sample-fps 5
```

Ese análisis guarda, por vídeo y por rango, un `frames.jsonl` con diagnóstico completo por frame, un `frames.csv` plano y un `summary.json` agregado en `football_ai/.data/partidoPrueba/pruebas/homography_analysis/`.

Para auditar visualmente casos representativos (`recovered_good`, `borderline_good`, `rejected_no_homography`, etc.) a partir de esos artefactos:

```bash
.venv/bin/python football_ai/pruebas/audit_homography_visual.py
```

Genera paneles con frame original, overlay de keypoints/líneas detectados, detecciones+ground points y bird-eye en `football_ai/.data/partidoPrueba/pruebas/homography_visual_audit/`.

Para renderizar un vídeo a pantalla partida con `el_clasico`: frame original + detecciones a la izquierda y campo 2D con círculos en `field_position_m` a la derecha:

```bash
.venv/bin/python football_ai/pruebas/render_split_homography_video.py
```

---

## 🛠️ Tecnologías

| Área | Tecnología |
|---|---|
| Detección | YOLOv8 / YOLOv11 (Ultralytics), fine-tuning con dataset Roboflow |
| Tracking | ByteTrack (supervision), extendido con restricción de equipo, doble señal de clase y posiciones 2D sobre el campo |
| Identificación de equipo | KMeans (scikit-learn), espacio de color LAB (OpenCV) |
| Evaluación | NumPy, pandas, Plotly, seaborn, matplotlib |
| Configuración | YAML (`config.yaml` centralizado) |
| Datasets | Roboflow (descarga automatizada) |

---

## 📦 Instalación

### Requisitos previos
- **Python 3.13.7** (entorno objetivo recomendado del repositorio)
- **Compatibilidad declarada del paquete:** `pyproject.toml` permite `>=3.8`
- **~10 GB de espacio libre** (modelos + datasets)
- **CUDA 12.4 + GPU NVIDIA** (opcional, para aceleración — CPU funciona pero es lento)

### Paso 1: Clonar el repositorio
```bash
git clone <repository-url>
cd narrador-futbol
```

### Paso 2: Crear entorno virtual limpio
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Verifica la versión activa (objetivo recomendado: 3.13.7)
python --version
```

### Paso 3: Actualizar pip e instalar PyTorch con CUDA support
```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar PyTorch 2.6.0 con soporte CUDA 12.4 (funciona en CPU también)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Si prefieres solo CPU (sin preparación para GPU futura):
# pip install torch torchvision
```

### Paso 3.1: (Opcional) Aceleración GPU de TeamDetector con RAPIDS
Si quieres acelerar el clustering LAB de camisetas (`TeamDetector`), puedes activar RAPIDS (`cuml-cu12`).

```bash
# Recomendado: evita usar /tmp si tiene poco espacio
mkdir -p ~/tmp_pip
TMPDIR=~/tmp_pip pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com cuml-cu12
```

Notas importantes:
- `cuml-cu12` descarga dependencias grandes (varios GB temporales).
- Si usas PyTorch con CUDA 12.4 (`cu124`), RAPIDS puede actualizar librerías CUDA a 12.9 y generar avisos de conflicto en pip.
- El código mantiene fallback automático a CPU si RAPIDS no está disponible o falla su importación.

### Paso 4: Instalar dependencias del proyecto
```bash
# Instalar todos los paquetes del requirements.txt
pip install -r requirements.txt

# Instalar el paquete 'football_ai' en modo editable (importante!)
pip install -e .
```

### Paso 5: Configurar variables de entorno
```bash
# Copia .env.example a .env solo si necesitas credenciales o overrides locales
cp .env.example .env

# Variables opcionales:
# ROBOFLOW_API_KEY=<tu_clave_aqui>          # solo para descargar datasets
# ELEVENLABS_API_KEY=<tu_clave_aqui>        # solo para ElevenLabs
# LLAMA_CPP_BASE_URL=http://127.0.0.1:8001 # si reutilizas un llama.cpp ya levantado
```

### Paso 6: Verificar la instalación
```bash
# Ejecuta el script de verificación
python verify_setup.py

# Debería mostrar 8/9 o 9/9 chequeos pasados (el .env needs API key es warning, no error)
```

### Paso 7: Descargar modelos y datos

El runtime por defecto no usa el YOLO base descargable. Usa el fine-tuned:

```text
models/finetuning/yolov11m/weights/best.pt
```

Ese archivo pesa ~39 MB y se puede versionar en GitHub sin Git LFS. El `.gitignore` ya permite trackear precisamente ese checkpoint si queréis dejarlo como artefacto oficial de la rama.

Opciones válidas:
- preferida para una rama reproducible: versionar `models/finetuning/yolov11m/weights/best.pt`
- alternativa: copiar ese checkpoint manualmente en esa ruta tras clonar
- fallback explícito: cambiar `paths.models.modelo_base` a otro peso válido

Los modelos base descargables siguen siendo útiles para pruebas, entrenamiento o para ese fallback, pero no son el detector por defecto del tracking actual.

```bash
# Descargar modelos YOLO base de referencia
python scripts/data/download_models.py

# Descargar dataset de detección desde Roboflow (opcional; requiere ROBOFLOW_API_KEY válida)
python scripts/data/download_datasets.py
```

Nota:
- `download_models.py` no deja listo por sí solo el runtime por defecto.
- Si no vais a versionar `models/finetuning/yolov11m/weights/best.pt`, hay que copiarlo manualmente o sobrescribir `paths.models.modelo_base`.

### Paso 7.1: Assets de runtime

#### PathCRF (`external/pathcrf`)

Con la configuración actual es obligatorio, porque `tracking.actions.enabled: true` y el pipeline espera ese checkout para la detección de acciones.

No basta con copiar solo el checkpoint a `models/`: el runtime actual importa módulos del repo externo (`models.utils`, `inference`, `datatools.postprocess`), lee `saved/<trial>/args.json` y carga el checkpoint dentro de ese trial. Por eso, hoy es mejor mantener `PathCRF` como repo externo clonado en `external/pathcrf`.

```bash
git clone <ruta-o-fork-de-pathcrf> external/pathcrf
```

La ruta configurada por defecto espera el trial en:

```text
external/pathcrf/saved/120/model/state_dict_best_acc.pt
```

#### `llama.cpp` local (`external/llama.cpp/config.yaml`)

Con la configuración actual es obligatorio salvo que sobrescribas `commentary.llm_base_url` o arranques la interfaz con otro backend explícito.

```bash
mkdir -p external/llama.cpp
cat > external/llama.cpp/config.yaml <<'YAML'
server:
  executable: /ruta/a/llama-server
  host: 127.0.0.1
  port: 8081
model:
  path: /ruta/al/modelo.gguf
YAML
```

Después ajusta:
- `server.executable` a tu binario `llama-server`
- `model.path` al GGUF real que quieras servir

Si prefieres no usar `external/llama.cpp`, puedes:
- usar `--commentary-backend ollama`, o
- exportar `LLAMA_CPP_BASE_URL=http://host:puerto`

#### Assets demasiado pesados para GitHub normal

Estos assets no conviene meterlos en git normal porque superan claramente los `100 MB` por archivo o dependen de repos externos:

- `models/pnlcalib/SV_kp` y `models/pnlcalib/SV_lines`
- `models/gguf/**/*.gguf`
- `external/pathcrf` completo

Estado actual recomendado:
- `PnLCalib`: el runtime clona `external/pnlcalib` y descarga automáticamente `SV_kp` y `SV_lines` en `models/pnlcalib/`
- `PathCRF`: mantenerlo como repo externo en `external/pathcrf`, porque el runtime necesita código + `args.json` + checkpoint, no solo pesos
- `GGUF`: descargarlo/manualmente o gestionarlo fuera del repo principal

### Paso 8: Descargar dataset sintético SoccerSynth (SpiideoSynLoc)
Para obtener un modelo YOLO más robusto, se recomienda pre-entrenarlo con el dataset sintético SoccerSynth. Su descarga es manual:
1. Regístrate en [research.spiideo.com](https://research.spiideo.com/).
2. Accede al apartado del dataset **Spiideo SoccerNet SynLoc**.
3. Acepta los términos y descarga manualmente los siguientes archivos `.zip` en la sección **FullHD Images**:
   - `train.zip`, `val.zip`, `test.zip`
   - `annotations.zip` (desde la sección Annotations)
4. Crea la carpeta `data/detection/SoccerSynth/SpiideoSynLoc` en este repositorio.
5. Mueve los `.zip` descargados a esa carpeta y descomprímelos todos ahí.

### Seleccionar intérprete en VS Code
1. **Ctrl+Shift+P** → **Python: Select Interpreter**
2. Si no aparece `.venv`, selecciona **"Enter interpreter path..."**
3. Escribe: `C:\Users\hecto\UNI\4\TFG\narrador-futbol\.venv\Scripts\python.exe`
4. Presiona Enter

---

## 🔍 Verificación de instalación

Siempre que hagas cambios en dependencias o uses el proyecto en una nueva terminal:

```bash
python verify_setup.py
```

Este script verifica:
- ✓ Python disponible (objetivo recomendado: 3.13.7)
- ✓ Venv activo
- ✓ Archivos de configuración (config.yaml, .env)
- ✓ Dependencias instaladas
- ✓ Módulos de football_ai importables
- ✓ PyTorch y estado de CUDA
- ✓ Variables de entorno configuradas

---

## 🧾 Interfaz de alineaciones

Ahora el proyecto incluye una interfaz web ligera en `interfaz/` para:

- introducir el nombre de cada equipo
- indicar el color de camiseta que se usará como referencia de equipo durante el tracking
- elegir la formación (`4-3-3`, `5-3-2`, `4-4-2`)
- escribir el jugador asociado a cada slot táctico
- elegir modo de comentarios `live` o `deferred` (por defecto `live`)
- lanzar `scripts/track.py` automáticamente con un `lineup_spec.json`
- arrancar automáticamente el servidor local de comentarios y precalentar un `intro` al abrir la interfaz
- intentar lanzar automáticamente el backend LLM configurado; por defecto usa `llama.cpp` leyendo `external/llama.cpp/config.yaml`
- dejar la interfaz disponible enseguida y mover el warmup de Gemma/XTTS a segundo plano
- bloquear el resto del formulario hasta que los dos nombres de equipo estén completos
- no rehacer las tarjetas mientras estás escribiendo el segundo equipo, para que el foco no se pierda a mitad de edición
- mostrar el MP4 final dentro de la propia interfaz cuando la ejecución termina

Ejecución:

```bash
python interfaz/app.py
```

Después abre:

```text
http://127.0.0.1:8767
```

La interfaz guarda un spec por ejecución en `output/interfaz/runs/<run_id>/lineup_spec.json` y llama a `scripts/track.py --lineup-spec ...`. El `<run_id>` ya no es un hash opaco: incluye fecha/hora local, equipos, vídeo y un sufijo corto, por ejemplo `20260413-153012_madrid-vs-barcelona_video-prueba-ajustado_a1b2`.
También deja el manifiesto de comentarios en `output/interfaz/runs/<run_id>/commentaries/events_manifest.jsonl`; en `live` intenta reproducir el `intro` precalentado nada más guardar y, cuando el tracking termina, ensambla una pista diferida desde ese manifiesto para incrustarla en el MP4 final. Además, cuando la ejecución nace desde la interfaz, el subprocess de tracking activa un bridge incremental `tracking -> PathCRF -> servidor de comentarios` solo para ese run, de modo que los scripts sueltos del repo siguen sin ejecutar PathCRF ni comentar jugadas automáticamente.
Ese bridge descarta reenvíos casi idénticos de PathCRF, guarda comentarios de texto aunque el audio esté ocupado y solo pide un nuevo WAV si la acción cae fuera de la ventana temporal ocupada por la generación + duración del audio anterior. Los eventos que ocurren mientras suena otro audio no quedan en cola sonora: permanecen en el manifiesto como texto. El flujo activo de acciones usa `football_ai/actions/rolling.py`: mantiene una ventana bounded, lanza checkpoints asíncronos y consolida eventos con `postprocess_emit_block`. Las acciones se etiquetan además con zona de campo (`iniciacion`, `creacion`, `finalizacion`) según el tercio del largo y la dirección de ataque del equipo; las acciones en finalización y los tiros tienen prioridad alta, y algunas acciones en iniciación pueden disparar un comentario de contexto generado por el LLM con datos tácticos de la alineación y clasificación simulada. Si ese comentario contextual está sonando y aparece un tiro o gol, el nuevo audio se marca como interrupción y la interfaz corta el comentario anterior. La pista diferida tampoco desplaza WAV antiguos hacia delante; omite los audios que se solaparían y, si un tiro/gol pisa un contexto interruptible, conserva la acción peligrosa y descarta ese contexto de la pista sonora. El servidor de comentarios sigue omitiendo duplicados consecutivos del mismo evento semántico básico (`action` + `player_name` + equipo). El MP4 final que sirve la interfaz se reexporta además como `H.264/AAC` y se limita a `1080p`, porque el tracking base seguía escribiendo `mp4v` y algunos navegadores mostraban en negro los exports demasiado grandes.
La interfaz intenta usar dos comentaristas cuando existen referencias de Qwen VoiceDesign cacheadas: mantiene XTTS como backend estable y elige entre la voz masculina y la femenina con aleatoriedad controlada, sin permitir mas de tres comentarios seguidos de la misma voz.
Si `--commentary-backend auto` no se toca, la interfaz intenta usar `llama.cpp` cuando existe `external/llama.cpp/config.yaml`; si no, cae a `ollama`. Si `--commentary-base-url` apunta a una URL local, la interfaz intenta arrancar ese backend localmente; si apuntas a un backend remoto, ese autoarranque no se intenta.
El fichero `external/llama.cpp/config.yaml` fija el binario `llama-server`, el alias expuesto por la API y el GGUF que se cargará al abrir la interfaz.
Ese precalentado ya no bloquea el arranque visible de la UI: la interfaz HTTP sube primero y el warmup de Gemma/XTTS sigue en segundo plano.

Cuando el tracker estabiliza los slots:

- el equipo se resuelve con los colores introducidos por el usuario, igual que en el tracking normal
- la formación seleccionada sustituye el once esperado fijo de `config.yaml` para esa ejecución
- si existe un slot único o ya desdoblado (`MC_IZQ`, `MC_DCHO`, `DC_IZQ`, `DC_DCHO`), se asigna también `player_name` al track y al resumen final

---

## 🎙️ Comentarios sintéticos

El proyecto incluye también un módulo en `football_ai/commentaries/` para convertir eventos ya detectados o simulados en comentarios cortos de narrador usando Ollama.

Ejemplo rápido:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}'
```

El modelo por defecto es `gemma4:e2b` con temperatura `0.7`. En general el comentario sale corto, debe incluir literalmente la accion del evento y el minuto solo se menciona en `gol`. El servidor recuerda el último comentario generado por tipo de acción y lo pasa al prompt para evitar repetir la misma frase o verbo principal. Para `gol`, el prompt deja ahora mas libertad para una narracion mas larga y emocionante. Tambien existen acciones especiales `intro` para abrir la retransmision sin jugador asociado y `contexto` para que el comentarista de apoyo aporte datos tácticos o clasificación simulada en zonas de bajo riesgo.

Ejemplo de apertura:

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"intro","event_time_s":0.0,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo"}' \
  --text-only
```

Si quieres evaluar solo el LLM sin pasar por TTS:

```bash
python -m football_ai.commentaries.eval_llm \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --show-prompts \
  --show-raw-response
```

Si quieres evaluar el Gemma cuantizado servido por `llama.cpp`:

```bash
python -m football_ai.commentaries.eval_llm \
  --backend llama_cpp \
  --base-url http://127.0.0.1:8001 \
  --model gemma4-q4ks-text \
  --event-json '{"action":"pase largo","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","field_zone":"medio campo","action_index":30}'
```

Ese mismo módulo puede convertir el comentario a audio. El backend estable sigue siendo XTTS, pero ahora también puedes probar dos rutas de Qwen:

- `qwen`: flujo Python `VoiceDesign -> Base`, primero diseña una voz de narrador y luego la reutiliza.
- `qwen_cpp`: runtime experimental con `qwen3-tts.cpp`, speaker embedding cacheado y modelos GGUF.

El informe técnico completo de todas las pruebas realizadas con Qwen, incluyendo tiempos reales, FlashAttention, `qwen3-tts.cpp`, el papel de `vLLM-Omni` y la decisión final, está en [football_ai/report/qwen_tts_evaluation_report.md](/home/cmantill/narrador-futbol/football_ai/report/qwen_tts_evaluation_report.md).

La salida del LLM pasa por una limpieza final que evita interjecciones exageradas tipo `GOOOOOOOOL` o palabras con letras estiradas de forma poco natural.
En GPU, el backend `qwen` intenta cargar el modelo siguiendo la ruta recomendada por la demo oficial de Qwen TTS, con `device_map="cuda:0"` y `flash_attention_2` cuando esta disponible.

```bash
python -m football_ai.commentaries \
  --event-json '{"action":"gol","player_name":"Bellingham","player_position":"MC","event_time_s":132.4,"team_name":"Real Madrid","opponent_team_name":"Wolfsburgo","field_zone":"frontal del area","action_index":30}' \
  --tts-backend qwen \
  --audio-out output/commentaries/audio/demo.wav
```

Para baja latencia, puedes mantener el backend TTS en caliente con un servidor HTTP local:

```bash
python -m football_ai.commentaries \
  --http-server \
  --tts-backend qwen \
  --model gemma4:e2b \
  --base-url http://127.0.0.1:11435
```

El servidor escucha por defecto en `http://127.0.0.1:8788` y acepta `POST /api/commentaries`.

También expone `POST /api/commentaries/stream` como SSE. Ese endpoint emite `accepted`, `commentary`, `tts_start` y `completed`, de modo que el cliente recibe el texto del LLM antes de que termine la síntesis del WAV.

Ejemplo con `qwen_cpp`:

```bash
python -m football_ai.commentaries \
  --http-server \
  --tts-backend qwen_cpp \
  --qwen-cpp-repo-dir /tmp/qwen3-tts.cpp \
  --qwen-cpp-model-dir output/commentaries/qwen_cpp_runtime/models \
  --qwen-cpp-threads 6 \
  --model gemma4:e2b \
  --base-url http://127.0.0.1:11435
```

Si necesitas volver al backend estable:

```bash
python -m football_ai.commentaries --tts-backend xtts
```

En esta maquina, la mejor configuracion Qwen para latencia sigue siendo `Qwen3-TTS-12Hz-0.6B-Base` por la ruta Python, sin FlashAttention y manteniendo el proceso vivo. El servidor hace warmup real antes de quedar listo, de modo que el arranque cuesta unos `14 s`, pero luego un `pase largo` ha bajado de ~`15.8 s` en ejecucion puntual a ~`7.6-9.0 s` por peticion en caliente. Cuando Qwen entra por GPU, el backend intenta cargar el modelo completo en VRAM sin offload, asi que lo normal es usar un unico worker persistente y no lanzar varios procesos Qwen a la vez.

La ruta `qwen_cpp` ya queda integrada y funcional. Ahora intenta construir `ggml` con CUDA y enlazar una build-wrapper propia de `qwen3-tts.cpp`; si esa ruta no entra en la maquina actual, cae a CPU y sigue funcionando. La primera pasada puede tardar bastante por la compilacion de `ggml-cuda`, pero despues reutiliza la libreria resultante. En la RTX 3080 de desarrollo, `6` hilos ha sido la configuracion mas rapida y estable en servidor persistente.

La decision de arquitectura actual para produccion es mantener `XTTS` como backend estable y reutilizar `Qwen VoiceDesign` solo para generar una voz de locutor de referencia. El codigo experimental de Qwen esta aislado en `football_ai/commentaries/experimental/` para no sobrecargar `football_ai/commentaries/voice.py`.

---

## 🚀 Uso

Todos los scripts se ejecutan desde la **raíz del proyecto**. La configuración se lee automáticamente de `config.yaml`.

### Pipeline completo de tracking
```bash
python scripts/track.py
```
También puedes indicar un shortcut de vídeo definido en `paths.data`:
```bash
python scripts/track.py video_prueba_ajustado
```
Para iteraciones rápidas de tracking también puedes usar el clip corto de 200 frames de YouTube:
```bash
python scripts/track.py video_yt_30s_200f
```
También puedes sobreescribir por terminal los colores de equipo y convertirlos a `LAB` de OpenCV automáticamente:
```bash
python scripts/track.py video_prueba_ajustado --team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"
```
Si quieres perfilar cuellos de botella por frame (sin alterar resultados), activa:
```bash
python scripts/track.py video_prueba_ajustado --profile-phases
```
  Esto imprime tiempos por fase y el total de cada frame.
Para benchmarks o ejecuciones aisladas, `scripts/track.py` también acepta overrides útiles:
```bash
python scripts/track.py video_prueba \
  --model-path models/finetuning/yolov11m/weights/best.pt \
  --output-root output/analysis/run_video_prueba_best \
  --experiment-label video_prueba__best \
  --execution-mode debug \
  --skip-render-video \
  --skip-metrics-dataset
```
- `--model-path`: usa un checkpoint distinto sin tocar `config.yaml`.
- `--output-root`: guarda `tracks.json`, `summary.json`, `debug_frames.json` y logs en un directorio dedicado.
- `--execution-mode debug`: activa las trazas ricas del pipeline y el guardado de `debug_frames.json`.
- `--skip-render-video`: evita renderizar el MP4 anotado final.
- `--skip-metrics-dataset`: no modifica `data/posiciones_etiquetadas/common/tracking_metrics.csv`.

Para lanzar el benchmark 4x3 pedido sobre `video_prueba`, `clasico_30s`, `ferro_30s` y `ucl_30s` con los tres modelos base/fine-tuned:
```bash
python scripts/run_tracking_model_benchmark.py
```
Ese runner ejecuta las 12 combinaciones llamando internamente a `scripts/track.py` y deja un árbol como este:
- `output/analysis/tracking_model_benchmark/<timestamp>/benchmark_manifest.json`
- `.../comparison_summary.csv`
- `.../comparison_summary.json`
- `.../runs/<video>__<modelo>/tracks.json`
- `.../runs/<video>__<modelo>/summary.json`
- `.../runs/<video>__<modelo>/debug_frames.json`
- `.../runs/<video>__<modelo>/stdout.log`
- `.../runs/<video>__<modelo>/stderr.log`

`--team-colors` acepta:
- lenguaje natural de color (ej. `rojo`, `verde clarito`, `azul marino`, `rojo oscuro`)
- HEX (ej. `#90EE90`)
- RGB (ej. `255,255,255`)
Los nombres de equipo se mapean de forma flexible contra los equipos de `config.yaml` (mayúsculas/minúsculas, acentos y pequeñas erratas).
Si pasas un nombre que no exista en `config.yaml` (ej. `Stutgart`), no falla: ese nombre se usa como equipo nuevo para esa ejecución.
Si en `--team-colors` hay al menos un equipo nuevo, se usan exactamente los equipos indicados ahí para ese run.
Si no quieres depender de nombres de equipo predefinidos, puedes activar bootstrap automático por color:
```bash
python scripts/track.py video_prueba_ajustado --team-mode auto-bootstrap --team-bootstrap-frames 1
```
En `auto-bootstrap`, el sistema aprende clusters de color al inicio del vídeo y fija esos equipos durante todo el run (nombres neutrales como `Equipo 1`, `Equipo 2`).
En este modo, el color de cada equipo se calcula como la **mediana por cluster** (más robusta a outliers que la media).
También se exige un mínimo de muestras por cluster (por defecto 4): si aparece un cluster pequeño (<=3), se re-clusteriza sobre el cluster grande.
Para los clips de `data/partidosPosiciones/` hay shortcuts `video_test_*` (ejemplo: `video_test_1`, `video_test_29`).
Si quieres lanzar tracking en lote con autodetección de fuente de vídeo (`--video-source auto`):
```bash
python scripts/track_partidos_posiciones.py
```
En `auto`, el script intenta primero el dataset Kaggle DFL (train+test) y, si no está disponible, cae a `data/partidosPosiciones`.
Si quieres forzar cada modo:
```bash
# Solo clips de data/partidosPosiciones
python scripts/track_partidos_posiciones.py --video-source partidos-posiciones

# Dataset Kaggle DFL completo (train+test)
python scripts/track_partidos_posiciones.py --video-source kaggle-all
```
Si el dataset ya está descargado localmente:
```bash
python scripts/track_partidos_posiciones.py --video-source kaggle-all --kaggle-path /ruta/al/dataset
```
Por defecto solo procesa vídeos que **todavía no tienen** `output/tracks_json/tracker/<video_sanitizado>_tracks.json` (evita reejecutar).
Si quieres reprocesar todo:
```bash
python scripts/track_partidos_posiciones.py --force
```
Cuando el lote usa `data/partidosPosiciones` (modo `partidos-posiciones` o fallback de `auto`), el script aplica un plan fijo de colores por vídeo (`video_test_*`) y nombres de equipo consistentes por color.
Ejemplo de convención:
- `blanco` -> `Real Madrid`
- `negro` -> `Equipo Negro`
- `rojo` -> `Equipo Rojo`
- `amarillo` -> `Equipo Amarillo`
- `gris` -> `Equipo Gris`
- `azul` -> `Equipo Azul`

Si quieres introducir colores manualmente en cada vídeo:
```bash
python scripts/track_partidos_posiciones.py --prompt-team-colors
```
Si prefieres bootstrap automático por color en cada vídeo:
```bash
python scripts/track_partidos_posiciones.py --video-source kaggle-all --team-mode auto-bootstrap --team-bootstrap-frames 1
```
Para `kaggle-all`, si no pasas `--team-colors` ni `--team-mode`, el batch usa `auto-bootstrap` automáticamente.
Modo prueba (sin ejecutar):
```bash
python scripts/track_partidos_posiciones.py --dry-run
```
También puedes pasar colores de equipo para todo el lote:
```bash
python scripts/track_partidos_posiciones.py --team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"
```
Si quieres desactivar el plan fijo por vídeo:
```bash
python scripts/track_partidos_posiciones.py --no-default-color-plan --team-colors "{Madrid:blanco, Wolsfburgo:verde-claro}"
```
Paralelización opcional del lote:
```bash
# CPU o pruebas controladas
python scripts/track_partidos_posiciones.py --workers 2 --continue-on-error

# Forzar paralelo en GPU (usar con cautela por VRAM)
python scripts/track_partidos_posiciones.py --workers 2 --allow-gpu-parallel --continue-on-error
```
Ejecuta detección + identificación de equipo + ByteTrack, genera el vídeo anotado en `output/` y muestra métricas en consola.
Por defecto usa `paths.data.video_prueba_corto` (si existe) y, en caso contrario, `paths.data.video_prueba`.
El MP4 de salida se guarda en `output/pruebaTracker/` con el nombre del vídeo de entrada y sufijo `_tracking.mp4` (ejemplo: `partido_ajustado_tracking.mp4`).
El JSON de tracks se guarda en `output/tracks_json/tracker/<video_sanitizado>_tracks.json` (formato esperado por `experiments/positions`) y además en `output/tracks_json/tracker/tracks.json` como compatibilidad legacy.
También se guarda el resumen por vídeo en `output/tracks_json/tracker/<video_sanitizado>_summary.json`.
Si el tracking online de roles está activo, `track.py` exporta además los CSV y PNG canónicos en `output/tracker/<video_sanitizado>_role_artifacts/`, incluyendo `<video_sanitizado>_frame_role_predictions.csv`, `<video_sanitizado>_player_role_summary.csv` y `<video_sanitizado>_greedy_role_diagnostics.csv`. La inferencia online usa el Set Transformer frame a frame con Hungarian por equipo contra `positions.expected_roles_by_team`; el voto limpio de cada frame se acumula por segmento y, para pintar el rol definitivo, cada frame vuelve a resolver otra asignación Hungarian usando la mayoría acumulada de los segmentos activos. Esos segmentos se reinician cuando hay señales fuertes de relink o saltos estructurales, y los cortes se reflejan en `tracks.json` con metadatos como `identity_segment_id`, `identity_reset_reason`, `canonical_assignment_mode` y `source_raw_tracker_id`. También deja una copia de compatibilidad en `output/tracks_json/tracker/`.
En identificación de equipos, el `bootstrap`/reclustering de colores sigue evaluándose cada 5 frames, pero la actualización pesada de clusters se limita a una sola vez por frame y por clase (`player`/`referee`) para evitar picos de latencia acumulados dentro del mismo frame.
Y se actualiza automáticamente un dataset acumulado de métricas de tracking en `data/posiciones_etiquetadas/common/tracking_metrics.csv` (una fila por vídeo, con upsert por `video_source`). Ese resumen incluye también `ball_coverage`, para medir en qué fracción del clip el balón quedó trackeado.
El tracking también calcula posesión online en la fase desacoplada `football_ai.posession` (`CANONICALTRACK -> POSESSION`), estimando equipo + jugador poseedor por frame. Ese dato se inyecta en los payloads de `tracks.json` (`ball_owning_team_id`, `ball_owning_player_id`, `player_id`, `is_possession_player`, `possession_reason`), se guarda también por frame en `tracks["possession"]` y se visualiza en el vídeo con un segundo recuadro amarillo en el jugador poseedor y un banner `POS: <equipo>`.
Cuando `tracking.use_field_positions=true`, cada frame se calibra con `PnLCalib` y el tracker usa coordenadas 2D reales del campo para `player` y `goalkeeper`, reduciendo el efecto del paneo de cámara en el matching.
La calibración adaptativa de `PnLCalib` ya no repite la inferencia de red en cada intento de thresholds: hace un único `forward` por frame, reutiliza esos heatmaps para reconstruir candidatos con cada par `keypoint_threshold`/`line_threshold` y solo reintenta la parte de decodificación, calibración y validación de calidad. El resultado externo se mantiene, pero baja el coste por frame cuando hay varios intentos adaptativos.
`FILTERING` ya entrega en `clean` solo las detecciones aceptadas; el detalle completo de aceptadas/rechazadas y sus `reject_code` queda en `trace`.
Si `canonical.reserve_penalty_spot_seed_players=true`, el tracker reserva además dos IDs canónicos sintéticos como `player` en los puntos de penalti. No participan en el clustering de equipos y solo sirven para que una detección real posterior pueda heredar esos IDs por geometría. Mientras no se absorban, también se escriben en el JSON con `synthetic_seed=true`.
Si `positions.special_seed_role_team_assignment_enabled=true`, el tracking principal ejecuta además el modelo de `position_role` frame a frame durante el tracking para los jugadores normales y usa a los defensas detectados en ese frame para asignar equipo a los IDs reservados `1-2` por defensa más cercano. Esos dos IDs no entran al Set Transformer: se etiquetan manualmente como `POR`. Si defines `positions.expected_roles_by_team`, ese once esperado sí se aplica frame a frame en la salida online mediante Hungarian por equipo, pero el histórico acumulado sigue guardando también la etiqueta cruda del modelo (`predicted_role_unconstrained`) para detectar swaps y evitar contaminar segmentos. Esos IDs no usan color de camiseta para recuperar identidad ni para fijar su equipo.
Para `player/goalkeeper` con homografía disponible, la reasignación canónica final usa exactamente el mismo gate de distancia en campo que ByteTrack (`field_position_match_distance_*`), sin suelo extra ni expansión por velocidad en la capa 2. Así un ID final no puede reaparecer con un salto mayor que el permitido en la capa base.
Si hay coordenadas de campo disponibles, el vídeo anotado muestra bajo cada `player` su posición `pos(m): x, y`.
Si en un frame `PnLCalib` falla (por ejemplo, homografía singular), el pipeline no aborta: ese frame se procesa con `field_position_m` no disponible y el tracking continúa.
En Linux headless, si `visualization.show_output=true` pero no hay `DISPLAY`/`WAYLAND_DISPLAY`, el sistema desactiva automáticamente la ventana de preview y continúa guardando el video de salida. El render final sale siempre en 4 paneles: en `tracking.execution_mode=runtime`, el panel de descartes queda negro porque no se construyen trazas internas; en `debug`, se rellenan los overlays completos y se guarda `debug_frames.json`.
Si otra persona ya tiene este repositorio clonado, le basta con hacer `git pull`; no tiene que clonar `PnLCalib` manualmente. En la primera ejecución, el código clona `PnLCalib` en `external/pnlcalib/` y descarga sus pesos automáticamente en `models/pnlcalib/`. Si no tiene este repositorio principal en local, entonces sí tiene que clonar `narrador-futbol` una vez antes de hacer `git pull` en el futuro.

### Detección básica
```bash
.venv/bin/python scripts/detect.py video_prueba yolo_v11_m
```

También acepta rutas desde la raíz del proyecto:

```bash
.venv/bin/python scripts/detect.py data/partidoPrueba/partido.mp4 models/yolo/v11/yolo11m.pt
```

Opcionalmente admite `--execution-mode runtime|debug`. `detect.py` usa por defecto `runtime`, pero reconstruye los overlays desde `clean` para seguir funcionando con el contrato nuevo de fases aunque no haya traza enriquecida.

Para generar detecciones más homografía PnLCalib en una sola pasada:

```bash
.venv/bin/python scripts/homography.py video_prueba_medio modelo_base
```

También acepta rutas desde la raíz del repo:

```bash
.venv/bin/python scripts/homography.py data/partidoPrueba/partido_medio.mp4 models/finetuning/yolov11m/weights/best.pt
```

La salida se guarda en `output/homography/<modelo>/<timestamp>/` con:
- `homography.json`: detecciones por frame + metadatos completos de homografía (`quality_diagnostics`, intentos, keypoints, líneas, score, rechazo, etc.)
- `homography.mp4`: vídeo a pantalla partida con detecciones a la izquierda y campo 2D a la derecha (incluye proyección de detecciones; rechazadas en rojo). La vista izquierda dibuja líneas/keypoints y la derecha oscurece la zona del campo fuera de la vista del frame.

`homography.py` ahora sigue el mismo flujo por fases que el tracker (`DetectionPhase -> ProjectionPhase -> FilteringPhase`) y por defecto usa `--execution-mode debug` para preservar la traza rica de homografía.

El script normaliza clases a `player`, `referee`, `ball` y `goalkeeper`, dibuja `bbox + confidence + clase` y guarda el MP4 anotado junto al JSON en `output/detect/<modelo>/<timestamp>/`.

## 🧪 Experimentos de geometría del campo

Para empezar a proyectar jugadores a coordenadas del campo hay un notebook de investigación en:

```bash
jupyter lab experiments/reference_points/reference_points.ipynb
```

Ese experimento implementa un enfoque clásico de visión por computador:
- segmentación del césped por color;
- extracción de líneas blancas con brillo, baja saturación y filtros para eliminar blobs compactos de jugadores;
- detección de segmentos con Hough y clustering por orientación;
- construcción de rectángulos candidatos a partir de pares de líneas del campo;
- estimación de homografía `imagen -> campo` y estabilización temporal frame a frame.

La longitud y anchura del campo están parametrizadas en el notebook para poder ajustar la plantilla a cada fuente de vídeo.

También hay una alternativa basada en modelo de keypoints del campo:

```bash
jupyter lab experiments/reference_points/soccana_keypoints.ipynb
```

Ese notebook descarga el modelo `Adit-jain/Soccana_Keypoint` desde Hugging Face, detecta 29 keypoints semánticos del campo y estima la homografía con RANSAC.

Y hay un tercer experimento basado en `PnLCalib`:

```bash
jupyter lab experiments/reference_points/pnlcalib_reference_points.ipynb
```

Ese notebook:
- clona `PnLCalib` bajo `external/pnlcalib/` si no existe;
- descarga los pesos `SV_kp` y `SV_lines` en `models/pnlcalib/`;
- detecta keypoints y líneas del campo con los modelos originales del repositorio;
- estima homografía `imagen -> campo`, hace warp a bird-eye y proyecta tracks al campo 2D reescalando sus coordenadas a la resolución real del frame usado por la homografía.

Igual que en `scripts/track.py`, no hace falta clonar `PnLCalib` a mano en otra máquina: el notebook lo descarga automáticamente la primera vez.

Importante: `PnLCalib` está publicado con licencia `GPL-2.0`, así que si este método se fuese a integrar en un producto cerrado habría que revisar esa implicación legal antes.

## 🧪 Dataset de roles posicionales (experimental)

El repositorio incluye un flujo experimental para crear un dataset supervisado de rol nominal de jugador (por ejemplo, `POR`, `DFC_IZQ`, `MC`, `DC`) a partir de:
- clips en `data/partidosPosiciones/`
- tracking con homografía (`field_position_m`) en `output/tracks_json/tracker/`
- etiquetado manual periódico por JSON cada 6 segundos (`frame_id + team_id + player_id`)

Entrada principal:
- Notebook: `experiments/positions/position_role_dataset.ipynb`
- Dataset/base features: `football_ai/positions/data/`
- Entrenamiento/inferencia sobre dataset común: `experiments/set_transformer.ipynb`
- API pública reusable: `football_ai/positions/model/`

En producción, `football_ai.positions` está organizado en `data/`, `model/` y `pipeline/`. `data/` separa observaciones, plantillas de etiquetado y dataset común; `model/` separa arquitectura, carga/splits, entrenamiento, inferencia y render; `pipeline/` concentra el motor online. El tracking usa `pipeline/online.py`, que aplica una doble pasada del algoritmo de Húngaro: primero genera votos limpios por frame y después resuelve el slot definitivo de cada segmento activo.

Salida del notebook (por ejecución):
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/base_table.csv`
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/samples_metadata_and_obj_features.csv`
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/samples_teammates.npz`
- `data/posiciones_etiquetadas/<match_id>_<timestamp>/dataset_meta.json`

Salida adicional acumulada (dataset común):
- `data/posiciones_etiquetadas/common/base_table.csv`
- `data/posiciones_etiquetadas/common/samples_metadata_and_obj_features.csv`
- `data/posiciones_etiquetadas/common/samples_teammates.npz`
- `data/posiciones_etiquetadas/common/dataset_meta.json`
- `data/posiciones_etiquetadas/common/sources.jsonl`
- `data/posiciones_etiquetadas/labels/<match_id>_labels_every_6s.json`

Ejecución:
```bash
jupyter lab experiments/positions/position_role_dataset.ipynb
```

## 🤖 Modelo Set Transformer para roles posicionales

Además del notebook de etiquetado, el repositorio incluye un notebook y un pipeline para entrenar un clasificador basado en Set Transformer a partir del dataset común acumulado en `data/posiciones_etiquetadas/common/base_table.csv` y aplicarlo después a `data/partidoPrueba/partido_ajustado.mp4`.

Antes de construir las features del modelo, cada equipo se lleva a una vista táctica canónica. Si un equipo ataca hacia `-x`, la representación se rota 180° a `(1 - x, 1 - y)` para conservar la semántica posicional `IZQ/DER` además de la profundidad del campo.

Entrenamiento por CLI:

```bash
python -m football_ai.positions.model.cli train
```

Si has cambiado la canonización/ingeniería de features y quieres ignorar el cache derivado antiguo:

```bash
python -m football_ai.positions.model.cli train --rebuild-from-base-table
```

Aplicación sobre `partido_ajustado`:

```bash
python -m football_ai.positions.model.cli predict \
  --model-path models/positions/set_transformer/<timestamp>/set_transformer_checkpoint.pt \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```

El notebook equivalente está en `experiments/set_transformer.ipynb` y ejecuta ese mismo flujo de forma interactiva. Soporta dos modos: reutilizar un checkpoint ya entrenado o reentrenar antes de predecir. Después de la predicción puede renderizar también el MP4 anotado con `role`. Además, fuerza una recarga explícita del módulo de entrenamiento para que los cambios recientes del pipeline se apliquen aunque el kernel de Jupyter siga vivo.

En el notebook puedes además fijar un once esperado por `team_id` y resolver la etiqueta estable con una asignación global tipo Hungarian. El modelo sigue produciendo probabilidades por jugador, pero el postproceso impone el multiconjunto de roles permitido para cada equipo; si faltan jugadores detectados, algunas plazas pueden quedar vacías, y si sobran jugadores detectados, los restantes caen a la mejor posición permitida dentro de ese once esperado.
Cuando esa restricción está activa, el vídeo anotado renderiza el rol estable restringido por equipo, no la etiqueta frame a frame libre del clasificador.

Artefactos generados:
- checkpoint y métricas en `models/positions/set_transformer/<timestamp>/`
- predicciones por frame en `output/predictions/positions/partido_ajustado_<timestamp>/frame_role_predictions.csv`
- resumen estable por jugador en `output/predictions/positions/partido_ajustado_<timestamp>/player_role_summary.csv`
- tracks enriquecidos con roles predichos en `output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json`
- vídeo anotado con roles en `output/predictions/positions/partido_ajustado_<timestamp>/partido_ajustado_roles_annotated.mp4`

Para renderizar el MP4 anotado a partir del JSON enriquecido:

```bash
python -m football_ai.positions.model.cli render-video \
  --video-path data/partidoPrueba/partido_ajustado.mp4 \
  --tracks-path output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json
```

Limitación actual:
- el dataset común etiquetado no contiene muestras `POR`, así que el pipeline asigna `POR` por heurística a los tracks cuya clase es `goalkeeper`.

## ⚽ Posesión de balón por equipo (experimental)

También hay un experimento incremental para estimar por frame qué equipo tiene la posesión del balón a partir del tracking ya generado en `output/tracks_json/tracker/`.

Entrada principal:
- Script: `experiments/possession/team_possession.py`
- Vídeo de prueba habitual: `data/partidoPrueba/partido_ajustado.mp4`

La heurística actual:
- toma el centro del bbox del balón;
- busca el jugador o portero más cercano usando el pie aproximado del bbox;
- confirma toque/control cuando la proximidad coincide con una caída de velocidad, un cambio de dirección, un control del mismo jugador o una recuperación clara del rival;
- añade histéresis temporal para no cambiar de equipo con una única observación rival dudosa;
- rellena lagunas cortas y elimina segmentos mínimos espurios para estabilizar la posesión mostrada;
- mantiene siempre la posesión del último equipo que tocó el balón entre toques y solo cambia de equipo si el balón muestra señal de toque real en velocidad y/o dirección.

Ejecución:

```bash
python -m experiments.possession.team_possession \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```

Artefactos generados:
- `output/predictions/possession/partido_ajustado_<timestamp>/frame_possession.csv`
- `output/predictions/possession/partido_ajustado_<timestamp>/summary.json`
- `output/predictions/possession/partido_ajustado_<timestamp>/partido_ajustado_possession_annotated.mp4`

### Fine-tuning del modelo
```bash
python scripts/train/finetune_player.py
```

### Adaptar `tracks.json` al formato PathCRF
```bash
python scripts/actions/convert_tracks_to_pathcrf.py output/tracks_json/tracker/partido_corto_tracks.json
```
Este paso genera un parquet ancho en `football_ai/actions/pathcrf/data/narrador/tracking_processed/` con 22 slots fijos de jugadores, 3 árbitros y variables de estado por frame. Si faltan tracks en algún frame, el adaptador interpola huecos internos y rellena ausencias persistentes con una plantilla simple de formación alineada al equipo visible.
En la versión actual del adaptador, las trayectorias exportadas se suavizan de forma más agresiva con mediana móvil, Savitzky-Golay y limitación de jitter por frame. Además, los slots PathCRF ya no se infieren por heurística espacial: salen directamente del reparto fijo de IDs canónicos (`1 -> home_1`, `2 -> away_1`, `3-12 -> home_2..11`, `13-22 -> away_2..11`, `23-25 -> referee_1..3`), se filtran seeds/observaciones sintéticas antes de recalcular velocidades y `player_id`/`ball_owning_team_id` se dejan vacíos por defecto en el parquet final para no contaminar PathCRF con una posesión heurística poco fiable.

### Ejecutar inferencia PathCRF sobre la salida del tracker

El vídeo dedicado de PathCRF ya no congela una etiqueta global por slot para todo el partido: cuando dispone de `tracks.json`, el overlay usa el `player_name` y `display_role_slot` del frame actual para evitar que un slot herede durante todo el render una mayoría histórica incorrecta.
El overlay del `tracking.mp4` sigue la misma prioridad semántica: si el frame ya tiene `display_role_slot`, se pinta esa salida final del segundo Hungarian de segmento antes que las mayorías intermedias del segmento. Así evitamos que el vídeo muestre un slot repetido que sí existe en `segment_majority_expected_role_slot` pero ya fue desambiguado en `display_role_slot`.
```bash
python scripts/actions/run_pathcrf.py video_prueba_corto
```

También acepta directamente el JSON ya exportado por `track.py`:

```bash
python scripts/actions/run_pathcrf.py output/tracks_json/tracker/partido_corto_tracks.json
```

El script:
- resuelve la salida de `track.py` desde el shortcut o desde la ruta que le pases;
- convierte el `tracks.json` a `*_tracking.parquet` si hace falta;
- carga el checkpoint de PathCRF del repo clonado en `external/pathcrf/` (por defecto `trial=120`, `state_dict_best_acc.pt`);
- exporta `*_edge_sequence.parquet`, `*_events.parquet`, `*_events_semantic.parquet`, `*_macro_prev.parquet`, `*_macro_next.parquet` y `*_summary.json` en `output/actions/pathcrf/<video>/`;
- si también dispone de `tracks.json`, genera además `*_commentary_events.json` con:
  - el mapeo invertido `pathcrf_id -> track_id`;
  - nombre del jugador, equipo, rival y posición cuando se pueden resolver desde el tracking enriquecido;
  - posición en metros, dirección de ataque estimada y zona de campo (`iniciacion`, `creacion`, `finalizacion`);
  - prioridad de comentario, dando más peso a finalización, tiro y gol;
  - un subpayload `commentary_event` listo para la fase posterior de Gemma;
  - un postproceso de posesión para saltar pases/controles cuyo actor ya no debería tener el balón y convertir pases a jugadores del rival en `robo` del receptor;
- genera además `*_pitch_pathcrf.mp4` con un drawer que, si conoce el vídeo original y el `tracks.json`, renderiza sobre el broadcast real usando las `bbox` reales e incrusta un mini-mapa 2D semitransparente en la esquina superior derecha; si no, cae al modo 2D puro.

Notas:
- el wrapper local soporta los checkpoints `set_*` incluidos en el repo clonado aunque la `venv` no tenga `torch_geometric`; si se quisiera usar un checkpoint `gat`, entonces sí habría que instalar esa dependencia;
- `ball_x/ball_y` se deja vacío de forma deliberada para no contaminar PathCRF con una proyección de balón poco fiable.
- el postproceso semántico actual añade dos capas encima de `detect_events`: reclasificación de inicios de episodio a `corner`, `throw_in` y `goalkick`, y una heurística de `shot` adaptada al flujo local basado en `kick/control/out`.

### Ejecutar pipeline live snapshots (aislado)
```bash
python scripts/actions/run_live_snapshots_pathcrf.py video_prueba_corto
```

Este modo reproduce el bridge por snapshots de forma aislada:
- recorre `tracks.json` frame a frame;
- genera snapshots acumulados cada `N` frames (`--snapshot-interval-frames`) tras warmup (`--min-frames`);
- en cada snapshot vuelve a ejecutar PathCRF legacy completo;
- deja resultados por snapshot y `live_snapshots_summary.json`.

### Ejecutar pipeline incremental (aislado)
```bash
python scripts/actions/run_incremental_pathcrf.py video_prueba_corto
```

Este modo ejecuta `ActionsRuntime` causal:
- actualiza estado en cada frame;
- corre inferencia PathCRF cada `N` frames (`--cadence-frames`) tras warmup (`--min-frames-warmup`);
- aplica confirmación/cooldown de acciones en línea;
- exporta `tracking.parquet`, `edge_sequence.parquet`, `events_semantic.parquet` y `runtime_checkpoints.json`.

### Comparar PathCRF legacy vs incremental
```bash
python scripts/actions/compare_pathcrf_modes.py output/tracks_json/tracker/partido_corto_tracks.json
```

Este script está pensado para depurar divergencias entre el adaptador offline histórico (`football_ai.actions`) y el flujo rolling actual (`football_ai.actions.rolling`).
En la rama incremental actual, el warmup intenta parecerse más al legacy en dos puntos que sesgaban mucho la comparación: la primera observación real de cada slot se ancla sin arrastrarla con la seed/template y `ball_x/ball_y` vuelve a exportarse vacío para no meter un balón sintético fijo en el centro.

Genera en `output/actions/pathcrf_compare/<video>/`:
- `offline/`: tracking parquet, edges y eventos semánticos del pipeline legacy;
- `incremental/`: tracking parquet, edges y eventos semánticos del pipeline incremental actual;
- `compare/`: diffs por frame/slot (`tracking_diff.parquet`, `edge_diff.parquet`) y error contra observaciones canónicas reales (`observation_error.parquet`, `observation_error_summary.json`);
- `checkpoints/`: replay de `legacy-on-snapshot` en los frames donde el runtime incremental inferiría, con un `comparison.json` por checkpoint y un resumen global.

Flags útiles:
- `--max-frames 200` para un smoke test corto;
- `--checkpoint-limit 5` para no generar demasiados snapshots;
- `--window-size-frames N` para reproducir una ventana incremental truncada;
- `--no-crf`, `--decode`, `--trial`, `--sample-freq` y `--window-seconds` para alinear exactamente el experimento con el checkpoint que quieras inspeccionar.

### Grid search de hiperparámetros del tracker
```bash
python scripts/track_experiments.py
```
Genera `output/pruebaTracker/tracks.json` con todos los experimentos para analizar con `experiments/visualization/experiments_comparator.ipynb`.

---

## ⚙️ Configuración

Toda la configuración está centralizada en `config.yaml`. Desde el refactor actual, la configuración visual queda separada por fase: `projector`, `bytetracker`, `posession`, `actions`, `commentary`, `canonical` y `positions` son bloques top-level; `tracking` se reserva para la orquestación general (`execution_mode`, `lineup_spec`, runtime). Los valores más relevantes a ajustar:

```yaml
paths:
  models:
    finetuned_player: "models/finetuning/yolov11m/weights/best.pt"
  data:
    video_prueba_ajustado: "data/partidoPrueba/partido_ajustado.mp4"
    video_prueba: "data/partidoPrueba/partido.mp4"
    video_test_1: "data/partidosPosiciones/test (1).mp4"

detection:
  conf_threshold: 0.01
  ball_min_conf: 0.0035

team_detector:
  team_color_model_conf:
    allow_referee_bootstrap_sampling_from_outfield: false  # permite enviar muestras outfield al bucket referee antes de cerrar su bootstrap
    referee_upper_bound_cap: 20.0  # cap maximo del upper_bound robusto del cluster referee
  shirt_detector_conf:
    n_init: 3
    batch_max_iter: 12
    batch_tol: 0.001

tracking:
  print_runtime_devices: true  # print del dispositivo efectivo real de PnLCalib y frame_hook (cuda/cpu)
  projector:
    constructor:
      pnl_refine: false
      repo_path: null
      weights_kp_path: null
      weights_line_path: null
      projection_quality_analyzer_conf:
        adaptive_min_visible_keypoints: 7
        adaptive_min_visible_lines: 1
        validation_score_threshold: 0.56
  new_track_active_overlap_iou: 0.2  # IoU máxima permitida contra tracks activos al crear un track nuevo
  new_track_unconfirmed_overlap_iou: 0.1  # IoU máxima permitida contra tracks unconfirmed al crear un track nuevo
  new_track_candidate_overlap_iou: 0.0  # IoU máxima permitida entre nuevos candidatos del mismo frame
  track_thresh: 0.15          # Confianza mínima para activar un track
  track_buffer: 90            # Frames que sobrevive un track sin ser visto
  match_thresh: 0.945         # IoU mínimo para asociar detección a track
  minimum_consecutive_frames: 5
  max_total_tracks: 25
  max_tracks_per_class:
    player: 22
    ball: 1
    referee: 3
  # Anti-ID-switch por clase y movimiento
  low_conf_threshold: 0.01
  track_activation_threshold: 0.10
  bbox_center_distance_gate_px: 90.0
  bbox_center_distance_gate_max_lost_frames: 4
  bbox_center_distance_gate_cap_px: 360.0
  bbox_height_ratio_threshold: 0.20
  bbox_width_ratio_threshold: 0.30
  lost_time_penalty_weight: 0.12
  lost_time_penalty_max_frames: 10
  field_position_match_distance_gate_m: 1.5
  field_position_match_distance_cap_m: 6.0
  field_position_match_distance_max_lost_frames: 10
  field_position_match_distance_decay_per_frame: 0.5
  reassign_motion_growth_cap_frames: 12
  strict_person_class_separation: true
  reserve_penalty_spot_seed_players: true
  reserve_penalty_spot_seed_match_distance_m: 12.0
  special_seed_role_team_assignment_enabled: true
  # La capa canónica preserva el mismo ID si ByteTrack mantiene el mismo
  # raw_tracker_id y la continuidad geométrica básica sigue siendo válida.
  special_seed_role_model_path: "models/positions/20260427_002133/best_model.pt"
  special_seed_canonical_ids: [1, 2]
  special_seed_defender_roles: ["LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT"]
  expected_roles_by_team:
    Real Madrid: ["POR", "LD", "LI", "DFC_DER", "DFC_IZQ", "MC", "MC", "MI", "MD", "DC", "DC"]
    Wolfsburgo: ["POR", "LD", "LI", "DFC_DER", "DFC_IZQ", "DFC_CENT", "MC", "MI", "MD", "DC", "DC"]
  referee_sideline_band_distance_m: 3.0
  # Los slots de árbitro quedan fijados por zona:
  # 23 -> árbitro central (fuera de la franja lateral y dentro del carril central de jugadores)
  # 24 -> linier de la banda superior
  # 25 -> linier de la banda inferior
  ball:
    expected_position_gate_px: 90.0
    expected_position_gate_growth_per_frame: 35.0
    expected_position_confidence_relax: 1.4
    size_ratio_per_frame: 1.8
    size_min_samples: 5
    size_std_factor: 3.0
    size_std_floor: 1.0
    max_reassign_lost_frames: 30
    high_conf_override: 0.6

teams:
  Real Madrid:
    color_lab_opencv: [255, 127, 127]
  Wolfsburgo:
    color_lab_opencv: [224, 77, 196]
```

---

## 🎯 Límites de tracking en Fase 1

Para reducir creación de IDs nuevos y mantener estabilidad en el tracking, el sistema usa límites por clase sobre la salida final `tracks`:

- `goalkeeper`: 2
- `player`: 20
- `ball`: 1
- `referee`: 3

Puntos importantes:

- La validación de límites se hace sobre `Tracker.get_tracks(...)`, no sobre el conteo crudo de detecciones YOLO por frame.
- Cuando se alcanza el máximo global de IDs visibles (`max_total_tracks`), se prioriza reasignar IDs previos compatibles antes de crear IDs nuevos.
- La reasignación mantiene coherencia por clase/equipo y aplica filtros de movimiento/cercanía para evitar saltos de identidad.
- En la capa canónica actual, los `goalkeeper` usan siempre los IDs `1-2` y los `player` se reparten en dos bloques fijos por equipo: `3-12` para un equipo y `13-22` para el otro, con máximo de 10 jugadores por bloque.

Parámetros relevantes de `TRACKER_CONF` (gestionados en `football_ai/tracking/tracker.py` y `football_ai/bytetrack/byte_tracker.py`):

- `max_total_tracks`
- `second_match_threshold`
- `unconfirmed_match_threshold`
- `low_conf_threshold`
- `track_activation_threshold`
- `reassign_motion_factor`
- `reassign_min_distance`
- `reassign_min_samples`
- `reassign_motion_growth_cap_frames` (solo aplica a clases sin homografía)
- `bbox_center_distance_gate_px`
- Valor baseline actual recomendado tras la auditoría larga de `ucl_30s`: `90.0`
- `bbox_center_distance_gate_max_lost_frames`
- `bbox_center_distance_gate_cap_px`
- `bbox_height_ratio_threshold`
- `bbox_width_ratio_threshold`
- `lost_time_penalty_weight`
- `lost_time_penalty_max_frames`
- `field_position_match_distance_gate_m`
- `field_position_match_distance_cap_m`
- `field_position_match_distance_max_lost_frames`
- `field_position_match_distance_decay_per_frame`
- `strict_person_class_separation`
- `special_seed_role_team_assignment_enabled`
- `special_seed_role_model_path`
- `special_seed_canonical_ids`
- `special_seed_defender_roles`
- `expected_roles_by_team`

Si sigues viendo cambios de ID en clips largos, ajusta en este orden:
1. Activa `strict_person_class_separation`.
2. Ajusta los gates en campo (`field_distance_gate_*`) para `player/goalkeeper`.
4. Baja `reassign_min_distance` (píxeles, para clases sin campo) o endurece `field_position_match_distance_*` si el problema está en `player/goalkeeper`.

---

## 📊 Métricas de evaluación

El módulo `football_ai/evaluation/` calcula automáticamente estas métricas por track:

| Métrica | Descripción |
|---|---|
| `coverage` | % de frames del vídeo en que el track fue visible |
| `fragments` | Número de interrupciones en el track |
| `mean_speed` | Velocidad media de movimiento (px/frame) |
| `team_flip_rate` | Tasa de cambios incorrectos de equipo asignado |
| `entropy` | Entropía de la distribución de equipos del track |
| `color_var` | Varianza del color de camiseta detectado a lo largo del tiempo |
| `bbox_size_cv` | Coeficiente de variación del tamaño del bounding box |

---

## 🧹 Limpieza del proyecto

```bash
python clean_project.py --all       # Limpieza completa
python clean_project.py --cache     # Solo caché de Python
python clean_project.py --output    # Solo vídeos y JSONs generados
python clean_project.py --datasets  # Solo datasets (regenerables)
python clean_project.py --models    # Solo modelos base (regenerables)
```

> ⚠️ Los modelos **fine-tuned** no se eliminan en ningún caso (requieren horas de entrenamiento).

---

## 🐛 Solución de problemas

**`FileNotFoundError: config.yaml`** — Ejecuta los scripts desde la raíz del proyecto, no desde el directorio del script.

**`Could not open video`** — Verifica que la ruta en `config.yaml → paths.data` es correcta y el archivo existe.

**`CUDA out of memory`** — Reduce el batch en `config.yaml → finetuning.batch` o añade `device='cpu'` al script.

**Imports fallando** — Asegúrate de que el entorno virtual está activado y `pip install -r requirements.txt` se ejecutó correctamente.

---

## ❓ Preguntas Frecuentes

### ¿Qué es `football_ai.egg-info/` y por qué aparece?

`football_ai.egg-info/` es una **carpeta de metadatos** generada automáticamente por `pip install -e .` (instalación en modo editable). Contiene:
- `METADATA`: Información del paquete (versión, dependencias, autor)
- `WHEEL`: Información de la distribución
- `RECORD`: Lista de archivos instalados
- `entry_points.txt`: Scripts ejecutables del paquete

**¿Necesito regenerarla?** No. Se regenera automáticamente cuando:
- Cambias `pyproject.toml` o `setup.py`
- Ejecutas `pip install -e .` de nuevo
- Cambias dependencias en `setup.py`

**Para cambios normales en código Python**, no necesitas hacer nada. El modo editable permite cambios sin reinstalar.

**Si quieres hacer una limpieza completa:**
```bash
rm -r football_ai.egg-info/      # Linux/Mac
rmdir /s football_ai.egg-info/   # Windows
pip install -e .                  # Regenera
```

---

### ¿Puedo instalar NVIDIA CUDA Toolkit 12.4 ahora?

**Sí, es recomendable.** PyTorch ya está preparado con soporte CUDA 12.4 (`torch 2.6.0+cu124`).

#### Pasos para instalar CUDA 12.4:

1. **Verifica tu GPU NVIDIA:**
   ```bash
   nvidia-smi        # Si funciona, tienes NVIDIA instalado
   ```

2. **Descarga CUDA Toolkit 12.4:**
   - Ir a: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - Seleccionar OS (Windows/Linux) y arquitectura
   - Descargar e instalar

3. **Descarga cuDNN (acelerador para redes neuronales):**
   - Ir a: https://developer.nvidia.com/cudnn
   - Descargar cuDNN para CUDA 12.4
   - Seguir instrucciones de instalación del archivo README

4. **Verifica que PyTorch detecta CUDA:**
   ```bash
   python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
   ```
   Debería mostrar: `CUDA disponible: True` y `CUDA version: 12.4`

5. **Benchmark (opcional):**
   ```bash
   python -c "import torch; t = torch.randn(10000, 10000, device='cuda'); print((t @ t).sum())"
   ```
   Debería ejecutarse rápidamente (GPU) en lugar de lentamente (CPU).

**Beneficio:** Los fine-tunings y detecciones serán **10-100x más rápidos**.

---

### ¿El venv se va a romper con actualizaciones?

No. El venv es **completamente independiente** del Python global:
- Actualizaciones de Windows o sistema no afectan
- Otros proyectos pueden usar otros venvs sin conflictos
- Si algo falla, simplemente `rm -r .venv` y crea uno nuevo

---

### ¿Cómo actualizar dependencias sin romper nada?

```bash
# Ver qué versiones hay disponibles
pip index versions ultralytics

# Actualizar una dependencia específica
pip install --upgrade ultralytics

# Actualizar todas las dependencias
pip install --upgrade -r requirements.txt

# Después, regenera el lock (recomendado):
pip freeze > requirements-lock.txt
```

---

## 📋 Mantenimiento del entorno

### Limpiar caché y archivos temporales
```bash
python clean_project.py --cache
```

### Verificar integridad periódicamente
```bash
python verify_setup.py
```

### Actualizar football_ai si editaste código
```bash
# No es necesario. El modo editable permite cambios inmediatos.
# Solo si cambiaste setup.py o pyproject.toml:
pip install -e .
```

---

## ✅ Validación mínima al cambiar código

Si cambias código ejecutable (especialmente en detección/tracking), ejecuta al menos el pipeline principal afectado y verifica que arranca sin excepción inicial.

Ejemplo:

```bash
python scripts/track.py
```

Si ejecutas en entorno headless, puedes dejar `show_output: true`: la visualización en ventana se desactiva sola cuando no hay display. Si quieres forzarlo desde configuración, usa `show_output: false`.

---

## 📝 Estado del proyecto

| Fase | Estado |
|---|---|
| Fase 1: Detección, tracking e identificación | ✅ Completada |
| Fase 2: Detección de acciones | 📅 Pendiente |
| Fase 3: Generación de comentarios (LLM) | 📅 Pendiente |
| Fase 4: Síntesis de voz | 📅 Pendiente |
| Fase 5: Pipeline en tiempo real | 📅 Pendiente |

---

## 👤 Autor

Héctor García y Carlos Mantilla  
Universidad Complutense de Madrid  
Trabajo de Fin de Grado — 2026
