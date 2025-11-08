from ultralytics import YOLO

ruta_modelo = "../../models/yolo/v11/yolov11m.pt"
ruta_data_yaml = "../../data/detection/FootBall-Detection-2/data.yaml"
output_dir = ruta_modelo.replace("yolo", "finetuning").replace(".pt", "")

def finetuning(ruta_modelo, ruta_data_yaml, epochs, batch, output_dir):
    model = YOLO(ruta_modelo)
    
    model.train(
        data=ruta_data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="finetuning",
        project=output_dir,
    )

if __name__ == "__main__":
    finetuning(ruta_modelo, ruta_data_yaml, 50, 16, output_dir)
