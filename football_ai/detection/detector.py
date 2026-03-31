from ultralytics import YOLO

class Detector:
    def __init__(self, model_path, conf=0.01, verbose=False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.verbose = verbose
    
    def detect(self, video, stream=True):
        return self.model.predict(video, stream=stream, conf=self.conf, verbose=self.verbose)
