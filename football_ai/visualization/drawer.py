import os
import cv2

class Drawer:
    def __init__(self, colors, default_color=(255, 255, 255)):
        self.colors = colors
        self.DEFAULT_COLOR = default_color

    def create_writer(self, video, output_path):
        """
        Creates a VideoWriter to save the processed video.
        
        Args:
            video: Path to the input video
            output_path: Path where the output video will be saved
            
        Returns:
            Tuple (VideoCapture, VideoWriter)
            
        Raises:
            ValueError: If the video path is empty
            FileNotFoundError: If the video does not exist
            RuntimeError: If the video cannot be opened or the writer cannot be created
        """
        if not video:
            raise ValueError("Video path is empty.")
        if not os.path.isfile(video):
            raise FileNotFoundError(f"Video not found: {video}")

        root, ext = os.path.splitext(output_path)
        if ext == "":
            output_path = root + ".mp4"

        try:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not create video writer for: {output_path}")
            
            return cap, writer
            
        except Exception as e:
            if 'cap' in locals() and cap is not None:
                cap.release()
            raise RuntimeError(f"Error creating video writer: {e}") from e
    
    def draw_detection(self, frame, class_name, data, color, track_id):
        """Draws a single detection on the frame."""
        x1, y1, x2, y2 = map(int, data["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} #{track_id}"
        distances = data.get("distances")
        team = data.get("team")
        if distances is not None and team is not None:
            d_str = ", ".join(f"{t}: {d:.1f}" for t, d in distances.items())
            label += f" [{team}] ({d_str})"

        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def draw_all_detections_in_frame(self, frame, class_name, class_tracks, frame_id):
        """Draws all detections of a class in a frame."""
        frame_data = class_tracks[frame_id]
        color = self.colors.get(class_name, self.DEFAULT_COLOR)
        for track_id, data in frame_data.items():
            self.draw_detection(frame, class_name, data, color, track_id)

    def draw_tracks(self, tracks, video, output_path, show=False, window_name="Tracking"):
        """
        Draws the tracks on the video and saves it.
        
        Args:
            tracks: Dictionary with tracks per class
            video: Path to the input video
            output_path: Path where the video with tracks will be saved
            show: If True, displays the video in real time
            window_name: Name of the display window
            
        Raises:
            RuntimeError: If an error occurs during video processing
        """
        cap = None
        out = None
        
        try:
            cap, out = self.create_writer(video, output_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_id >= num_frames:
                    break
                
                for class_name, class_tracks in tracks.items():
                    if frame_id < len(class_tracks):
                        self.draw_all_detections_in_frame(frame, class_name, class_tracks, frame_id)
                
                out.write(frame)
                if show:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_id += 1

        except Exception as e:
            raise RuntimeError(f"Error drawing tracks: {e}") from e
        
        finally:
            # Release resources
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if show:
                cv2.destroyAllWindows()
