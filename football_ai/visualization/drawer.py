import os
import sys
import logging
import cv2

logger = logging.getLogger(__name__)


def _format_role_overlay_label(value):
    token = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    token = "_".join(part for part in token.split("_") if part)
    if token in {"MC_IZQ", "MC_DCHO", "DC_IZQ", "DC_DCHO"}:
        return token
    return str(value)

class Drawer:
    def __init__(self, colors, default_color=(255, 255, 255)):
        self.colors = colors
        self.DEFAULT_COLOR = default_color

    @staticmethod
    def _can_show_gui():
        """
        Checks whether a graphical display is available for OpenCV windows.

        Returns:
            True if UI windows can be shown safely, False otherwise.
        """
        if sys.platform != "linux":
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

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
        if data.get("synthetic_seed"):
            return
        if not isinstance(data.get("bbox"), (list, tuple)) or len(data["bbox"]) < 4:
            return
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

        info_lines = []

        predicted_role_frame = data.get("predicted_role_frame")
        predicted_role = data.get("predicted_role")
        assignment_method = data.get("assignment_method")
        expected_role_slot = data.get("expected_role_slot")
        display_role_slot = data.get("display_role_slot")
        constrained_expected_role = (
            bool(assignment_method) or bool(expected_role_slot) or bool(display_role_slot)
        )

        if constrained_expected_role and display_role_slot:
            info_lines.append(f"role: {_format_role_overlay_label(display_role_slot)}")
        elif constrained_expected_role and predicted_role:
            info_lines.append(f"role: {predicted_role}")
        elif predicted_role_frame and predicted_role:
            if str(predicted_role_frame) == str(predicted_role):
                info_lines.append(f"role: {predicted_role_frame}")
            else:
                info_lines.append(
                    f"role: {predicted_role_frame} | stable: {predicted_role}"
                )
        elif predicted_role_frame:
            info_lines.append(f"role: {predicted_role_frame}")
        elif predicted_role:
            info_lines.append(f"role: {predicted_role}")

        # Show field coordinates (meters) under player bbox when available.
        if class_name == "player":
            field_position = data.get("field_position_m")
            if (
                isinstance(field_position, (list, tuple))
                and len(field_position) >= 2
                and field_position[0] is not None
                and field_position[1] is not None
            ):
                info_lines.append(
                    f"pos(m): {field_position[0]:.1f}, {field_position[1]:.1f}"
                )

        for idx, info_label in enumerate(info_lines):
            baseline_y = min(y2 + 15 + (idx * 14), frame.shape[0] - 5)
            cv2.putText(
                frame,
                info_label,
                (x1, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

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
        show_window = show
        if show and not self._can_show_gui():
            logger.warning(
                "show=True pero no hay entorno gráfico (DISPLAY/WAYLAND). "
                "Se desactiva la visualización en tiempo real y solo se guardará el video de salida."
            )
            show_window = False
        
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
                if show_window:
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
            if show_window:
                cv2.destroyAllWindows()
