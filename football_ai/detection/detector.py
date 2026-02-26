from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, conf):
        self.model = YOLO(model_path)
        self.conf = conf
    
    def detect(self, video, stream=True):
        return self.model.predict(video, stream=stream, conf=self.conf, verbose=True)
