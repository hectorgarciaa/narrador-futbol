from ultralytics import YOLO


def normalize_detection_class_name(class_name):
    token = class_name.strip().lower()
    aliases = {
        "person": "player",
        "persons": "player",
        "people": "player",
        "human": "player",
        "player": "player",
        "players": "player",
        "goalkeeper": "goalkeeper",
        "goalkeepers": "goalkeeper",
        "gk": "goalkeeper",
        "keeper": "goalkeeper",
        "goalie": "goalkeeper",
        "referee": "referee",
        "referees": "referee",
        "ref": "referee",
        "refs": "referee",
        "arbitro": "referee",
        "arbitros": "referee",
        "árbitro": "referee",
        "árbitros": "referee",
        "ball": "ball",
        "balls": "ball",
        "sports ball": "ball",
        "sports balls": "ball",
    }
    return aliases.get(token, token)


class Detector:
    def __init__(self, model_path, conf=0.01, verbose=False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.verbose = verbose

    @staticmethod
    def _normalize_result(result):
        result.names = {
            class_id: normalize_detection_class_name(class_name)
            for class_id, class_name in result.names.items()
        }
        return result

    def detect(self, video, stream=True):
        results = self.model.predict(video, stream=stream, conf=self.conf, verbose=self.verbose)
        if stream:
            return (self._normalize_result(result) for result in results)
        return [self._normalize_result(result) for result in results]
