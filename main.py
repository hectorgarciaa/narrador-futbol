from ultralytics import YOLO
from roboflow import Roboflow
import supervision as sv
import numpy as np
import cv2
video_path = "08fd33_4.mp4"

model = YOLO("yolo11m.pt")  # load a pretrained YOLOv11m model

results = model.track(
    source=video_path,
    tracker="botsort.yaml",   # Usa ByteTrack como algoritmo
    show=False,                  # Mostrar video en tiempo real
    save=True                   # Guardar salida
)

# --- TRACKER (supervision) ---
byte_tracker = sv.ByteTrack(track_thresh=0.4, match_thresh=0.8, track_buffer=60)

# --- ANOTADOR VISUAL ---
box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default(), thickness=2, text_scale=0.6)

# --- CONFIGURACIÓN DE EQUIPOS ---
TEAM_COLORS = {
    "Real Madrid": np.array([0, 0, 255]),     # blanco
    "Wolfsburgo": np.array([70, 150, 150])    # verde
}

def get_avg_color_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    return np.array([h, s, v])

def classify_team(color_hsv):
    distances = {team: np.linalg.norm(color_hsv - ref) for team, ref in TEAM_COLORS.items()}
    return min(distances, key=distances.get)

# --- VIDEO ---
video_path = "08fd33_4.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("output_sv_tracking.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

player_teams = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- DETECCIÓN YOLO ---
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # --- APLICAR TRACKER ---
    detections = byte_tracker.update_with_detections(detections)

    # --- ASIGNAR EQUIPO (solo una vez por track_id) ---
    for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
        if track_id is None:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        if track_id not in player_teams:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            h_crop = crop.shape[0]
            shirt = crop[: int(h_crop * 0.5), :]
            color = get_avg_color_hsv(shirt)
            team = classify_team(color)
            player_teams[track_id] = team
            print(f"Jugador {track_id} -> {team}")

    # --- ANOTAR FRAME ---
    labels = [
        f"{player_teams.get(track_id, 'Desconocido')} #{int(track_id) if track_id else -1}"
        for track_id in detections.tracker_id
    ]

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

    # --- GUARDAR FRAME ---
    out.write(annotated_frame)
    cv2.imshow("tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video guardado como output_sv_tracking.mp4")
'''
rf = Roboflow(api_key="KSNZNjhrcZN1cq1zmo33")
project = rf.workspace("yolo-atnlh").project("football-detection-wfhdh")
dataset = project.version(2).download("yolov11")

# 2. Entrenar con tu dataset de Roboflow
model.train(
    data="football-detection-wfhdh-2/data.yaml",  # ruta al data.yaml
    epochs=50,           # número de épocas
    imgsz=640,           # tamaño de las imágenes
    batch=16,            # tamaño del batch
    name="football_finetune",  # nombre del experimento (opcional)
)

# Cargar el modelo ajustado
model = YOLO("runs/train/football_finetune/weights/best.pt")

# Detectar o hacer tracking en tu video
results = model.track(source="08fd33_4.mp4", show=False, save=True)
'''