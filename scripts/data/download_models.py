from ultralytics import YOLO
import os

carpeta_modelos = "../models/yolo"

# Listado de modelos a descargar
modelos = {
    "v8": {
        "yolov8m": "yolov8m.pt",
        "yolov8l": "yolov8l.pt",
        "yolov8x": "yolov8x.pt"
    },

    "v11": {
        "yolo11m": "yolov11m.pt",
        "yolo11l": "yolov11l.pt",
        "yolo11x": "yolov11x.pt"
    }
}

def downloadYoloModels(carpeta_modelos, modelos):
    # Descargar y guardar cada modelo
    for version, models in modelos.items():
        for nombre, archivo in models.items():
            modelo = YOLO(nombre)
            ruta_guardado = os.path.join(carpeta_modelos, version, archivo)
            modelo.save(ruta_guardado)
            print(f"{nombre} guardado en {ruta_guardado}")

if __name__ == "__main__":
    downloadYoloModels(carpeta_modelos, modelos)
