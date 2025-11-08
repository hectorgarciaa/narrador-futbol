from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, ruta_modelo):
        self.modelo = YOLO(ruta_modelo)
        self.tracker = sv.ByteTrack()

    def detect(self, partido):
        return self.modelo.predict(partido, stream=True, conf=0.1)
    
    def get_tracks(self, partido):
        detections = self.detect(partido)
        print(detections)
        for frame, detection in enumerate(detections):
            clases = detection.names
    

ruta_modelo = "../../../models/finetuning/v11/yolov11m/weights/best.pt"
partido = "../../../data/partidoPrueba/08fd33_4.mp4"

if __name__ == "__main__":
    t = Tracker(ruta_modelo)
    t.get_tracks(partido)
