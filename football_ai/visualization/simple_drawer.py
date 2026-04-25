import cv2


class SimpleDrawer:
    def __init__(self):
        self.colors = {
            "player": (0, 255, 0),
            "referee": (255, 0, 0),
            "ball": (0, 165, 255),
            "goalkeeper": (0, 255, 255),
        }
        self.letters = {
            "player": "P",
            "referee": "R",
            "goalkeeper": "G",
        }

    def draw(self, frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            color = self.colors[detection["class"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{detection["confidence"]:.2f}'
            if detection["class"] != "ball":
                label = f'{self.letters[detection["class"]]} {label}'
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame
