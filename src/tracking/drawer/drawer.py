import os
import cv2

class Drawer:
    def __init__(self, colors, default_color=(255, 255, 255)):
        self.colors = colors
        self.DEFAULT_COLOR = default_color
        pass

    def createWriter(self, video, output_path):
        if not video:
            raise ValueError("Video path is empty.")
        if not os.path.isfile(video):
            raise FileNotFoundError(f"Video not found: {video}")

        root, ext = os.path.splitext(output_path)
        if ext == "":
            output_path = root + ".mp4"

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        return cap, cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    def drawDetection(self, frame, class_name, data, color, track_id):
        x1, y1, x2, y2 = map(int, data["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} #{track_id}"
        distances = data.get("distances")
        team = data.get("team")
        if distances is not None and team is not None:
            label += "\n"
            d = [distances["Real Madrid"].round(1), distances["Wolfsburgo"].round(1)]
            label += f"{team}: {d}"

        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def drawAllDetectionsInFrame(self, frame, class_name, class_tracks, frame_id):
        frame_data = class_tracks[frame_id]
        color = self.colors.get(class_name, self.DEFAULT_COLOR)
        for track_id, data in frame_data.items():
            self.drawDetection(frame, class_name, data, color, track_id)

    def draw_tracks(self, tracks, video, output_path, show=False, window_name="Tracking"):
        cap, out = self.createWriter(video, output_path)

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_id >= num_frames:
                break
            
            for class_name, class_tracks in tracks.items():
                self.drawAllDetectionsInFrame(frame, class_name, class_tracks, frame_id)
            
            out.write(frame)
            if show:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_id += 1

        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()
