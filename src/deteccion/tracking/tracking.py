from ultralytics import YOLO
import supervision as sv
import os
import cv2

class Tracker:
    def __init__(self, ruta_modelo, conf_model, track_thresh, track_buffer, match_thresh, frame_rate):
        self.modelo = YOLO(ruta_modelo)
        self.conf_model = conf_model
        self.tracker = sv.ByteTrack(track_thresh, track_buffer, match_thresh, frame_rate)

    def detect(self, partido):
        return self.modelo.predict(partido, stream=True, conf=self.conf_model)

    
    def get_tracks(self, partido):
        detections = self.detect(partido)
        tracks = { "player": [], "goalkeeper": [], "referee": [], "ball": [] }

        for num_frame, detection_frame in enumerate(detections):
            detection_sv = sv.Detections.from_ultralytics(detection_frame)
            track = self.tracker.update_with_detections(detection_sv)
            
            tracks["player"].append({}), tracks["goalkeeper"].append({}), tracks["referee"].append({}), tracks["ball"].append({})

            for object_detected in track:
                bbox, _, confidence, class_id, tracker_id, class_name = object_detected
                class_name = class_name["class_name"]

                tracks[class_name][num_frame][tracker_id] = {"bbox": bbox, "confidence": confidence}

        return tracks
    
    def draw_tracks(self, partido, tracks, output_path):
        cap = cv2.VideoCapture(partido)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_idx = 0
        colors = { "player": (0, 255, 0), "goalkeeper": (0, 255, 255), "referee": (255, 0, 0), "ball": (0, 0, 255) }

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(tracks["player"]):
                break

            # Dibujar cajas para cada clase
            for class_name, frames_data in tracks.items():
                frame_data = frames_data[frame_idx]
                color = colors.get(class_name, (255, 255, 255))
                
                for track_id, data in frame_data.items():
                    x1, y1, x2, y2 = map(int, data["bbox"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name} #{track_id}", 
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 1, cv2.LINE_AA)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"✅ Video anotado guardado en: {output_path}")
    

ruta_modelo = "../../../models/finetuning/v11/yolov11m/weights/best.pt"
partido = "../../../data/partidoPrueba/08fd33_4.mp4"
output = "../../../output/pruebaTracker/08fd33_4_3.mp4"

if __name__ == "__main__":
    t = Tracker(ruta_modelo, 0.05, 0.05, 150, 1, 25)
    tracks = t.get_tracks(partido)
    t.draw_tracks(partido, tracks, output)

