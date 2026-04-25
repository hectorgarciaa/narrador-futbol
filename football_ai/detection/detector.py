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
    return aliases.get(token, None)


class Detector:
    def __init__(self, model_path, conf=0.01, verbose=False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.verbose = verbose

    @staticmethod
    def _normalize_result(result):
        normalized_names = {}
        kept_class_ids = set()

        for class_id, class_name in result.names.items():
            normalized = normalize_detection_class_name(class_name)
            if normalized is not None:
                normalized_names[class_id] = normalized
                kept_class_ids.add(int(class_id))

        result.names = normalized_names

        if result.boxes is None or len(result.boxes) == 0:
            return result

        keep_indices = [
            idx
            for idx, class_id in enumerate(result.boxes.cls.tolist())
            if int(class_id) in kept_class_ids
        ]

        if len(keep_indices) != len(result.boxes):
            result.boxes = result.boxes[keep_indices]
        
        return result

    def detect(self, video, stream=True):
        results = self.model.predict(video, stream=stream, conf=self.conf, verbose=self.verbose)
        if stream:
            return (self._normalize_result(result) for result in results)
        return [self._normalize_result(result) for result in results]
