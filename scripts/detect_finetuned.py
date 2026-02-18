from ultralytics import YOLO

ruta_modelo = "../../models/finetuning/v11/yolov11m/weights/best.pt"
ruta_partido = "../../data/partidoPrueba/08fd33_4.mp4"
ruta_salida = "../../output/"

def detect(ruta_modelo, ruta_partido, ruta_salida):
    model = YOLO(ruta_modelo)
    results = model(ruta_partido, save=True, project=ruta_salida, name="pruebaFinetuning", exist_ok=True)

if __name__ == "__main__":
    detect(ruta_modelo, ruta_partido, ruta_salida)
