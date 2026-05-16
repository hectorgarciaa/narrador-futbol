import cv2
import numpy as np


DEFAULT_CLASS_COLORS = {
    "player": (0, 255, 0),
    "referee": (255, 0, 0),
    "ball": (0, 165, 255),
    "goalkeeper": (0, 255, 255),
}


def panel_color(class_name, default=(255, 255, 255)):
    return DEFAULT_CLASS_COLORS.get(class_name, default)


class BasePanel:
    def __init__(self, size_hw, background=(0, 0, 0)):
        self.height, self.width = size_hw
        self.background = background

    def empty(self):
        panel = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        panel[:] = self.background
        return panel

    def draw_text_box(self, panel, lines, origin=(16, 22), color=(255, 255, 255), scale=0.5):
        x_coord, y_coord = origin
        for line in lines or []:
            if not line:
                continue
            cv2.putText(
                panel,
                str(line),
                (x_coord, y_coord),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )
            y_coord += int(22 * scale / 0.5)


class VideoPanel(BasePanel):
    def render(self, data):
        frame = data.get("frame")
        if frame is None:
            panel = self.empty()
        else:
            panel = np.asarray(frame).copy()
            if panel.shape[:2] != (self.height, self.width):
                panel = cv2.resize(panel, (self.width, self.height))
        for item in data.get("items", []):
            self._draw_item(panel, item)
        self.draw_text_box(panel, data.get("text_lines"))
        return panel

    def _draw_item(self, panel, item):
        bbox = item.get("bbox")
        if bbox is None:
            return
        x1, y1, x2, y2 = [int(value) for value in bbox]
        color = tuple(item.get("color") or (255, 255, 255))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, int(item.get("thickness", 2)))
        text_y = max(20, y1 - 8)
        for line in self._item_lines(item):
            cv2.putText(
                panel,
                line,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            text_y += 18

    def _item_lines(self, item):
        if item.get("lines") is not None:
            return [str(line) for line in item.get("lines", []) if line]
        lines = []
        class_name = item.get("class_name")
        confidence = item.get("confidence")
        if class_name is not None or confidence is not None:
            token = str(class_name or "")
            if confidence is not None:
                token = f"{token} {float(confidence):.2f}".strip()
            if token:
                lines.append(token)
        if item.get("team") is not None:
            lines.append(f"team: {item['team']}")
        if item.get("team_distances"):
            lines.append(f"dist: {item['team_distances']}")
        if self._valid_point(item.get("field_position_m")):
            x_coord, y_coord = item["field_position_m"][:2]
            lines.append(f"field: {float(x_coord):.1f}, {float(y_coord):.1f}")
        class_labels = [value for value in item.get("class_labels", []) if value]
        if class_labels:
            lines.append("cls: " + " | ".join(map(str, class_labels)))
        roles = [value for value in item.get("roles", []) if value]
        if roles:
            lines.append("roles: " + " | ".join(map(str, roles)))
        if item.get("player_name") is not None:
            lines.append(str(item["player_name"]))
        track_ids = item.get("track_ids")
        if isinstance(track_ids, dict):
            pairs = [f"{key}={value}" for key, value in track_ids.items() if value is not None]
            if pairs:
                lines.append("ids: " + " ".join(pairs))
        elif track_ids:
            lines.append("ids: " + " ".join(str(value) for value in track_ids if value is not None))
        lines.extend(str(line) for line in item.get("extra_lines", []) if line)
        return lines

    @staticmethod
    def _valid_point(point):
        return point is not None and len(point) >= 2 and np.all(np.isfinite(point[:2]))


class FieldPanel(BasePanel):
    def __init__(self, size_hw, background=(30, 110, 30), padding=24):
        super().__init__(size_hw, background=background)
        self.padding = int(padding)

    def render(self, data):
        panel = self.empty()
        field_length_m, field_width_m = data.get("field_size_m", (105.0, 68.0))
        self._draw_pitch(panel, field_length_m, field_width_m)
        shade_polygon = data.get("shade_outside_polygon_m")
        if shade_polygon:
            self._shade_outside_polygon(panel, shade_polygon, field_length_m, field_width_m)
        for item in data.get("lines", []):
            self._draw_field_line(panel, item, field_length_m, field_width_m)
        for item in data.get("points", []):
            self._draw_field_circle(panel, item, field_length_m, field_width_m)
        for item in data.get("entities", []):
            self._draw_field_circle(panel, item, field_length_m, field_width_m)
        self.draw_text_box(panel, data.get("text_lines"))
        return panel

    def _draw_pitch(self, panel, field_length_m, field_width_m):
        top_left = self._to_px((0.0, 0.0), field_length_m, field_width_m)
        bottom_right = self._to_px((field_length_m, field_width_m), field_length_m, field_width_m)
        center_top = self._to_px((field_length_m * 0.5, 0.0), field_length_m, field_width_m)
        center_bottom = self._to_px((field_length_m * 0.5, field_width_m), field_length_m, field_width_m)
        center = self._to_px((field_length_m * 0.5, field_width_m * 0.5), field_length_m, field_width_m)
        cv2.rectangle(panel, top_left, bottom_right, (255, 255, 255), 2)
        cv2.line(panel, center_top, center_bottom, (255, 255, 255), 2)
        cv2.circle(panel, center, max(12, int(min(panel.shape[:2]) * 0.08)), (255, 255, 255), 2)

    def _draw_field_circle(self, panel, item, field_length_m, field_width_m):
        position = item.get("position_m")
        if not self._valid_point(position):
            return
        center = self._to_px(position, field_length_m, field_width_m)
        color = tuple(item.get("color") or (255, 255, 255))
        radius = int(item.get("radius", 8))
        cv2.circle(panel, center, radius, color, -1)
        if item.get("canonical_id") is not None:
            cv2.putText(
                panel,
                str(item["canonical_id"]),
                (center[0] - 6, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        if item.get("position_label"):
            cv2.putText(
                panel,
                str(item["position_label"]),
                (center[0] - 18, center[1] + radius + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_field_line(self, panel, item, field_length_m, field_width_m):
        point_1 = item.get("point_1_m")
        point_2 = item.get("point_2_m")
        if not self._valid_point(point_1) or not self._valid_point(point_2):
            return
        cv2.line(
            panel,
            self._to_px(point_1, field_length_m, field_width_m),
            self._to_px(point_2, field_length_m, field_width_m),
            tuple(item.get("color") or (255, 255, 255)),
            int(item.get("thickness", 2)),
        )

    def _shade_outside_polygon(self, panel, polygon_m, field_length_m, field_width_m):
        polygon_px = [
            self._to_px(point, field_length_m, field_width_m)
            for point in polygon_m
            if self._valid_point(point)
        ]
        if len(polygon_px) < 3:
            return
        mask = np.zeros(panel.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon_px, dtype=np.int32)], 255)
        outside = mask == 0
        if not np.any(outside):
            return
        shade_color = np.asarray((10, 60, 10), dtype=np.uint8)
        alpha = 0.55
        panel[outside] = (
            panel[outside].astype(np.float32) * (1.0 - alpha)
            + shade_color.astype(np.float32) * alpha
        ).astype(np.uint8)

    def _to_px(self, point, field_length_m, field_width_m):
        usable_width = max(1, self.width - (2 * self.padding))
        usable_height = max(1, self.height - (2 * self.padding))
        x_coord = int(round(self.padding + (float(point[0]) * usable_width / max(float(field_length_m), 1.0))))
        y_coord = int(round(self.padding + (float(point[1]) * usable_height / max(float(field_width_m), 1.0))))
        return x_coord, y_coord

    @staticmethod
    def _valid_point(point):
        return point is not None and len(point) >= 2 and np.all(np.isfinite(point[:2]))


class VideoOutput:
    def __init__(self, output_path, fps, panels):
        self.output_path = output_path
        self.fps = float(fps)
        self.panels = panels
        panel_height = panels[0][0].height
        panel_width = panels[0][0].width
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (panel_width * len(panels[0]), panel_height * len(panels)),
        )

    def render_frame(self, panel_data):
        rows = []
        for row_index, row_panels in enumerate(self.panels):
            rendered_row = [
                panel.render(panel_data[row_index][col_index])
                for col_index, panel in enumerate(row_panels)
            ]
            rows.append(np.concatenate(rendered_row, axis=1))
        return np.concatenate(rows, axis=0)

    def write_frame(self, panel_data):
        self.writer.write(self.render_frame(panel_data))

    def close(self):
        self.writer.release()
