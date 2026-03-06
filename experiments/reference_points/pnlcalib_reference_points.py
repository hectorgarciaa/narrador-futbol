from __future__ import annotations

import importlib
import json
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

try:
    from experiments.reference_points.classical_reference_points import (
        PitchGeometry,
        build_pitch_template,
        find_project_root,
        get_video_frame_count,
        iterate_video_frames,
        load_default_video_path,
        load_tracks_json,
        resize_frame,
    )
except ModuleNotFoundError:
    from dataclasses import dataclass as _fallback_dataclass

    @_fallback_dataclass(frozen=True)
    class PitchGeometry:
        field_length_m: float = 106.0
        field_width_m: float = 68.0
        goal_width_m: float = 7.32
        penalty_area_depth_m: float = 16.5
        penalty_area_width_m: float = 40.32
        goal_area_depth_m: float = 5.5
        goal_area_width_m: float = 18.32
        penalty_mark_distance_m: float = 11.0
        center_circle_radius_m: float = 9.15
        penalty_arc_radius_m: float = 9.15
        corner_arc_radius_m: float = 1.0
        line_width_m: float = 0.12

        @property
        def halfway_x_m(self) -> float:
            return self.field_length_m / 2.0

        @property
        def center_y_m(self) -> float:
            return self.field_width_m / 2.0

        @property
        def penalty_area_top_y_m(self) -> float:
            return (self.field_width_m - self.penalty_area_width_m) / 2.0

        @property
        def penalty_area_bottom_y_m(self) -> float:
            return (self.field_width_m + self.penalty_area_width_m) / 2.0

        @property
        def goal_area_top_y_m(self) -> float:
            return (self.field_width_m - self.goal_area_width_m) / 2.0

        @property
        def goal_area_bottom_y_m(self) -> float:
            return (self.field_width_m + self.goal_area_width_m) / 2.0

    def find_project_root(start_path: Optional[Path] = None) -> Path:
        path = Path(start_path or Path.cwd()).resolve()
        for candidate in (path,) + tuple(path.parents):
            if (candidate / "config.yaml").exists():
                return candidate
        raise FileNotFoundError("No se ha encontrado config.yaml desde la ruta actual.")

    def load_default_video_path(
        project_root: Optional[Path] = None,
        preferred_keys: Sequence[str] = ("video_prueba_ajustado", "video_prueba"),
    ) -> Path:
        root = find_project_root(project_root)
        with open(root / "config.yaml", "r", encoding="utf-8") as file_handle:
            config = yaml.safe_load(file_handle)
        for key in preferred_keys:
            relative_path = config.get("paths", {}).get("data", {}).get(key)
            if relative_path:
                candidate = (root / relative_path).resolve()
                if candidate.exists():
                    return candidate
        raise FileNotFoundError("No se ha encontrado ningún vídeo de prueba configurado.")

    def get_video_frame_count(video_path: Path) -> int:
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return total_frames

    def iterate_video_frames(
        video_path: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
    ):
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        final_frame = total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
        frame_step = max(1, int(step))
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        frame_index = int(start_frame)
        while frame_index <= final_frame:
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                break
            yield frame_index, frame_bgr
            for _ in range(frame_step - 1):
                ok, _ = capture.read()
                if not ok:
                    capture.release()
                    return
            frame_index += frame_step
        capture.release()

    def resize_frame(frame_bgr: np.ndarray, max_width: int = 1280) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        if width <= max_width:
            return frame_bgr.copy()
        scale = max_width / float(width)
        return cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def load_tracks_json(tracks_path: Path) -> Dict[str, object]:
        with open(tracks_path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    def build_pitch_template(
        field_length_m: float = 106.0,
        field_width_m: float = 68.0,
        pixels_per_meter: int = 8,
    ):
        geometry = PitchGeometry(field_length_m=field_length_m, field_width_m=field_width_m)
        width_px = int(round(geometry.field_length_m * pixels_per_meter))
        height_px = int(round(geometry.field_width_m * pixels_per_meter))
        pitch_rgb = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        pitch_rgb[:, :] = (47, 128, 53)
        line_color = (240, 240, 240)
        thickness = max(1, int(round(geometry.line_width_m * pixels_per_meter)))

        cv2.rectangle(pitch_rgb, (0, 0), (width_px - 1, height_px - 1), line_color, thickness)
        halfway_x = int(round(geometry.halfway_x_m * pixels_per_meter))
        center_y = int(round(geometry.center_y_m * pixels_per_meter))
        cv2.line(pitch_rgb, (halfway_x, 0), (halfway_x, height_px - 1), line_color, thickness)
        cv2.circle(
            pitch_rgb,
            (halfway_x, center_y),
            int(round(geometry.center_circle_radius_m * pixels_per_meter)),
            line_color,
            thickness,
        )
        cv2.circle(pitch_rgb, (halfway_x, center_y), max(2, thickness), line_color, -1)

        def _draw_box(x_start_m, x_end_m, y_top_m, y_bottom_m):
            x1 = int(round(x_start_m * pixels_per_meter))
            x2 = int(round(x_end_m * pixels_per_meter))
            y1 = int(round(y_top_m * pixels_per_meter))
            y2 = int(round(y_bottom_m * pixels_per_meter))
            cv2.rectangle(pitch_rgb, (x1, y1), (x2, y2), line_color, thickness)

        _draw_box(
            0.0,
            geometry.penalty_area_depth_m,
            geometry.penalty_area_top_y_m,
            geometry.penalty_area_bottom_y_m,
        )
        _draw_box(
            geometry.field_length_m - geometry.penalty_area_depth_m,
            geometry.field_length_m,
            geometry.penalty_area_top_y_m,
            geometry.penalty_area_bottom_y_m,
        )
        _draw_box(
            0.0,
            geometry.goal_area_depth_m,
            geometry.goal_area_top_y_m,
            geometry.goal_area_bottom_y_m,
        )
        _draw_box(
            geometry.field_length_m - geometry.goal_area_depth_m,
            geometry.field_length_m,
            geometry.goal_area_top_y_m,
            geometry.goal_area_bottom_y_m,
        )

        return pitch_rgb, None, None, None, None


PNLCALIB_REPO_URL = "https://github.com/mguti97/PnLCalib"
PNLCALIB_REPO_DIRNAME = "pnlcalib_repo"
PNLCALIB_WEIGHTS = {
    "SV_kp": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
    "SV_lines": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
}
PNLCALIB_REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "sympy": "sympy",
    "ellipse": "lsq-ellipse",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "torch": "torch",
    "torchvision": "torchvision",
}


@dataclass
class PnLCalibRuntime:
    repo_dir: Path
    weights_kp_path: Path
    weights_line_path: Path
    device: str
    model_kp: torch.nn.Module
    model_line: torch.nn.Module
    resize_transform: T.Resize
    framebyframe_calib_cls: Any
    get_keypoints_from_heatmap_batch_maxpool: Any
    get_keypoints_from_heatmap_batch_maxpool_l: Any
    complete_keypoints: Any
    coords_to_dict: Any


@dataclass
class PnLCalibEstimate:
    keypoints_dict: Dict[int, Dict[str, float]]
    lines_dict: Dict[int, Dict[str, float]]
    camera_result: Optional[Dict[str, Any]]
    ground_result: Optional[Dict[str, Any]]
    homography_image_to_field: Optional[np.ndarray]
    homography_image_to_template: Optional[np.ndarray]
    projection_matrix: Optional[np.ndarray]
    reprojection_error: Optional[float]
    estimation_mode: str
    timing_detail: Dict[str, float] = field(default_factory=dict)

    @property
    def visible_keypoints_count(self) -> int:
        return len(self.keypoints_dict)

    @property
    def visible_lines_count(self) -> int:
        return len(self.lines_dict)

    @property
    def has_homography(self) -> bool:
        return self.homography_image_to_field is not None


def get_default_pnlcalib_repo_path(project_root: Optional[Path] = None) -> Path:
    root = find_project_root(project_root)
    return root / "models" / "reference_points" / PNLCALIB_REPO_DIRNAME


def get_default_pnlcalib_weights_dir(project_root: Optional[Path] = None) -> Path:
    root = find_project_root(project_root)
    return root / "models" / "reference_points" / "pnlcalib_weights"


def get_default_pnlcalib_weight_paths(
    project_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    weights_dir = get_default_pnlcalib_weights_dir(project_root)
    return weights_dir / "SV_kp", weights_dir / "SV_lines"


def get_video_frame_size(video_path: Path) -> Tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return width, height


def get_missing_pnlcalib_modules() -> Dict[str, str]:
    missing: Dict[str, str] = {}
    for module_name, package_name in PNLCALIB_REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing[module_name] = package_name
    return missing


def ensure_pnlcalib_repo(
    repo_dir: Optional[Path] = None,
    repo_url: str = PNLCALIB_REPO_URL,
) -> Path:
    target_dir = get_default_pnlcalib_repo_path() if repo_dir is None else Path(repo_dir)
    if target_dir.exists():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        check=True,
    )
    return target_dir


def ensure_pnlcalib_weights(
    weights_kp_path: Optional[Path] = None,
    weights_line_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    default_kp_path, default_line_path = get_default_pnlcalib_weight_paths()
    resolved_kp_path = default_kp_path if weights_kp_path is None else Path(weights_kp_path)
    resolved_line_path = default_line_path if weights_line_path is None else Path(weights_line_path)

    resolved_kp_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_line_path.parent.mkdir(parents=True, exist_ok=True)

    if not resolved_kp_path.exists():
        urllib.request.urlretrieve(PNLCALIB_WEIGHTS["SV_kp"], resolved_kp_path)
    if not resolved_line_path.exists():
        urllib.request.urlretrieve(PNLCALIB_WEIGHTS["SV_lines"], resolved_line_path)

    return resolved_kp_path, resolved_line_path


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _import_pnlcalib_modules(repo_dir: Path) -> Dict[str, Any]:
    _prepend_sys_path(repo_dir)
    importlib.invalidate_caches()

    cls_hrnet = importlib.import_module("model.cls_hrnet")
    cls_hrnet_l = importlib.import_module("model.cls_hrnet_l")
    utils_calib = importlib.import_module("utils.utils_calib")
    utils_heatmap = importlib.import_module("utils.utils_heatmap")

    return {
        "get_cls_net": cls_hrnet.get_cls_net,
        "get_cls_net_l": cls_hrnet_l.get_cls_net,
        "FramebyFrameCalib": utils_calib.FramebyFrameCalib,
        "get_keypoints_from_heatmap_batch_maxpool": utils_heatmap.get_keypoints_from_heatmap_batch_maxpool,
        "get_keypoints_from_heatmap_batch_maxpool_l": utils_heatmap.get_keypoints_from_heatmap_batch_maxpool_l,
        "complete_keypoints": utils_heatmap.complete_keypoints,
        "coords_to_dict": utils_heatmap.coords_to_dict,
    }


def load_pnlcalib_runtime(
    repo_dir: Optional[Path] = None,
    weights_kp_path: Optional[Path] = None,
    weights_line_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> PnLCalibRuntime:
    missing_modules = get_missing_pnlcalib_modules()
    if missing_modules:
        missing_description = ", ".join(
            f"{module} -> pip install {package}"
            for module, package in sorted(missing_modules.items())
        )
        raise ImportError(
            f"Faltan dependencias para PnLCalib: {missing_description}"
        )

    resolved_repo_dir = ensure_pnlcalib_repo(repo_dir)
    resolved_kp_path, resolved_line_path = ensure_pnlcalib_weights(weights_kp_path, weights_line_path)
    imported = _import_pnlcalib_modules(resolved_repo_dir)

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = resolved_repo_dir / "config" / "hrnetv2_w48.yaml"
    cfg_l_path = resolved_repo_dir / "config" / "hrnetv2_w48_l.yaml"

    with open(cfg_path, "r", encoding="utf-8") as file_handle:
        cfg = yaml.safe_load(file_handle)
    with open(cfg_l_path, "r", encoding="utf-8") as file_handle:
        cfg_l = yaml.safe_load(file_handle)

    loaded_state_kp = torch.load(resolved_kp_path, map_location=resolved_device)
    loaded_state_line = torch.load(resolved_line_path, map_location=resolved_device)

    model_kp = imported["get_cls_net"](cfg)
    model_kp.load_state_dict(loaded_state_kp)
    model_kp.to(resolved_device)
    model_kp.eval()

    model_line = imported["get_cls_net_l"](cfg_l)
    model_line.load_state_dict(loaded_state_line)
    model_line.to(resolved_device)
    model_line.eval()

    return PnLCalibRuntime(
        repo_dir=resolved_repo_dir,
        weights_kp_path=resolved_kp_path,
        weights_line_path=resolved_line_path,
        device=resolved_device,
        model_kp=model_kp,
        model_line=model_line,
        resize_transform=T.Resize((540, 960)),
        framebyframe_calib_cls=imported["FramebyFrameCalib"],
        get_keypoints_from_heatmap_batch_maxpool=imported["get_keypoints_from_heatmap_batch_maxpool"],
        get_keypoints_from_heatmap_batch_maxpool_l=imported["get_keypoints_from_heatmap_batch_maxpool_l"],
        complete_keypoints=imported["complete_keypoints"],
        coords_to_dict=imported["coords_to_dict"],
    )


def _projection_from_cam_params(cam_params: Dict[str, Any]) -> np.ndarray:
    x_focal_length = float(cam_params["x_focal_length"])
    y_focal_length = float(cam_params["y_focal_length"])
    principal_point = np.asarray(cam_params["principal_point"], dtype=np.float64)
    position_meters = np.asarray(cam_params["position_meters"], dtype=np.float64)
    rotation = np.asarray(cam_params["rotation_matrix"], dtype=np.float64)

    it_matrix = np.eye(4, dtype=np.float64)[:-1]
    it_matrix[:, -1] = -position_meters
    intrinsics = np.array(
        [
            [x_focal_length, 0.0, principal_point[0]],
            [0.0, y_focal_length, principal_point[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsics @ (rotation @ it_matrix)


def _homography_from_camera_result(camera_result: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if camera_result is None:
        return None

    cam_params = camera_result["cam_params"]
    intrinsics = np.array(
        [
            [cam_params["x_focal_length"], 0.0, cam_params["principal_point"][0]],
            [0.0, cam_params["y_focal_length"], cam_params["principal_point"][1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(cam_params["rotation_matrix"], dtype=np.float64)
    position = np.asarray(cam_params["position_meters"], dtype=np.float64)
    translation = -(rotation @ position.reshape(3, 1))
    homography_centered_to_image = intrinsics @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation[:, 0]]
    )
    try:
        homography_image_to_centered = np.linalg.inv(homography_centered_to_image)
    except np.linalg.LinAlgError:
        return None
    return homography_image_to_centered / homography_image_to_centered[2, 2]


def _centered_to_field_homography(
    homography_image_to_centered: Optional[np.ndarray],
    geometry: PitchGeometry,
) -> Optional[np.ndarray]:
    if homography_image_to_centered is None:
        return None

    centered_to_field = np.array(
        [
            [1.0, 0.0, geometry.field_length_m / 2.0],
            [0.0, 1.0, geometry.field_width_m / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homography_image_to_field = centered_to_field @ homography_image_to_centered
    return homography_image_to_field / homography_image_to_field[2, 2]


def _to_template_homography(
    homography_image_to_field: np.ndarray,
    pixels_per_meter: float = 8.0,
) -> np.ndarray:
    return np.array(
        [
            [pixels_per_meter, 0.0, 0.0],
            [0.0, pixels_per_meter, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ) @ homography_image_to_field


def estimate_pnlcalib_frame(
    frame_bgr: np.ndarray,
    runtime: PnLCalibRuntime,
    geometry: Optional[PitchGeometry] = None,
    keypoint_threshold: float = 0.3434,
    line_threshold: float = 0.7867,
    pnl_refine: bool = True,
    pixels_per_meter: int = 8,
) -> PnLCalibEstimate:
    total_start = perf_counter()
    if geometry is None:
        geometry = PitchGeometry()

    preprocess_color_start = perf_counter()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    preprocess_color_s = perf_counter() - preprocess_color_start

    preprocess_tensor_start = perf_counter()
    frame_tensor = TF.to_tensor(Image.fromarray(frame_rgb)).float().unsqueeze(0)
    preprocess_tensor_s = perf_counter() - preprocess_tensor_start

    _, _, original_height, original_width = frame_tensor.size()
    preprocess_resize_s = 0.0
    if frame_tensor.size()[-1] != 960:
        preprocess_resize_start = perf_counter()
        frame_tensor = runtime.resize_transform(frame_tensor)
        preprocess_resize_s = perf_counter() - preprocess_resize_start

    preprocess_to_device_start = perf_counter()
    frame_tensor = frame_tensor.to(runtime.device)
    preprocess_to_device_s = perf_counter() - preprocess_to_device_start
    _, _, resized_height, resized_width = frame_tensor.size()

    with torch.no_grad():
        model_kp_start = perf_counter()
        heatmaps_kp = runtime.model_kp(frame_tensor)
        model_kp_inference_s = perf_counter() - model_kp_start

        model_line_start = perf_counter()
        heatmaps_line = runtime.model_line(frame_tensor)
        model_line_inference_s = perf_counter() - model_line_start

    decode_keypoints_start = perf_counter()
    keypoint_coords = runtime.get_keypoints_from_heatmap_batch_maxpool(
        heatmaps_kp[:, :-1, :, :]
    )
    keypoints_batch = runtime.coords_to_dict(keypoint_coords, threshold=keypoint_threshold)
    decode_keypoints_s = perf_counter() - decode_keypoints_start

    decode_lines_start = perf_counter()
    line_coords = runtime.get_keypoints_from_heatmap_batch_maxpool_l(
        heatmaps_line[:, :-1, :, :]
    )
    lines_batch = runtime.coords_to_dict(line_coords, threshold=line_threshold)
    decode_lines_s = perf_counter() - decode_lines_start

    complete_keypoints_start = perf_counter()
    keypoints_dict, lines_dict = runtime.complete_keypoints(
        keypoints_batch[0],
        lines_batch[0],
        w=resized_width,
        h=resized_height,
        normalize=True,
    )
    complete_keypoints_s = perf_counter() - complete_keypoints_start

    calibrator_init_start = perf_counter()
    calibrator = runtime.framebyframe_calib_cls(
        iwidth=original_width,
        iheight=original_height,
        denormalize=True,
    )
    calibrator_init_s = perf_counter() - calibrator_init_start

    calibrator_update_start = perf_counter()
    calibrator.update(keypoints_dict, lines_dict)
    calibrator_update_s = perf_counter() - calibrator_update_start

    ground_voting_start = perf_counter()
    ground_result = calibrator.heuristic_voting_ground(refine_lines=pnl_refine)
    ground_voting_s = perf_counter() - ground_voting_start

    camera_voting_start = perf_counter()
    camera_result = calibrator.heuristic_voting(refine_lines=pnl_refine)
    camera_voting_s = perf_counter() - camera_voting_start

    homography_image_to_centered = None
    estimation_mode = "no_solution"
    reprojection_error: Optional[float] = None
    used_ground_solution = 0.0
    used_camera_fallback = 0.0

    homography_select_start = perf_counter()
    if ground_result is not None:
        homography_image_to_centered = np.asarray(ground_result["homography"], dtype=np.float64)
        reprojection_error = float(ground_result["rep_err"])
        estimation_mode = "ground_plane"
        used_ground_solution = 1.0
    else:
        homography_image_to_centered = _homography_from_camera_result(camera_result)
        if camera_result is not None:
            reprojection_error = float(camera_result["rep_err"])
            estimation_mode = "camera_fallback"
            used_camera_fallback = 1.0
    homography_select_s = perf_counter() - homography_select_start

    centered_to_field_start = perf_counter()
    homography_image_to_field = _centered_to_field_homography(
        homography_image_to_centered,
        geometry,
    )
    centered_to_field_s = perf_counter() - centered_to_field_start

    homography_image_to_template = None
    template_homography_s = 0.0
    if homography_image_to_field is not None:
        template_homography_start = perf_counter()
        homography_image_to_template = _to_template_homography(
            homography_image_to_field,
            pixels_per_meter=pixels_per_meter,
        )
        template_homography_s = perf_counter() - template_homography_start

    projection_matrix = None
    projection_matrix_s = 0.0
    if camera_result is not None:
        projection_matrix_start = perf_counter()
        projection_matrix = _projection_from_cam_params(camera_result["cam_params"])
        projection_matrix_s = perf_counter() - projection_matrix_start

    timing_detail = {
        "pnl_total_s": float(perf_counter() - total_start),
        "pnl_preprocess_color_convert_s": float(preprocess_color_s),
        "pnl_preprocess_tensor_build_s": float(preprocess_tensor_s),
        "pnl_preprocess_resize_s": float(preprocess_resize_s),
        "pnl_preprocess_to_device_s": float(preprocess_to_device_s),
        "pnl_model_kp_inference_s": float(model_kp_inference_s),
        "pnl_model_line_inference_s": float(model_line_inference_s),
        "pnl_decode_keypoints_s": float(decode_keypoints_s),
        "pnl_decode_lines_s": float(decode_lines_s),
        "pnl_complete_keypoints_s": float(complete_keypoints_s),
        "pnl_calibrator_init_s": float(calibrator_init_s),
        "pnl_calibrator_update_s": float(calibrator_update_s),
        "pnl_ground_voting_s": float(ground_voting_s),
        "pnl_camera_voting_s": float(camera_voting_s),
        "pnl_homography_select_s": float(homography_select_s),
        "pnl_centered_to_field_s": float(centered_to_field_s),
        "pnl_template_homography_s": float(template_homography_s),
        "pnl_projection_matrix_s": float(projection_matrix_s),
        "pnl_visible_keypoints_count": float(len(keypoints_dict)),
        "pnl_visible_lines_count": float(len(lines_dict)),
        "pnl_used_ground_solution": float(used_ground_solution),
        "pnl_used_camera_fallback": float(used_camera_fallback),
    }

    return PnLCalibEstimate(
        keypoints_dict=keypoints_dict,
        lines_dict=lines_dict,
        camera_result=camera_result,
        ground_result=ground_result,
        homography_image_to_field=homography_image_to_field,
        homography_image_to_template=homography_image_to_template,
        projection_matrix=projection_matrix,
        reprojection_error=reprojection_error,
        estimation_mode=estimation_mode,
        timing_detail=timing_detail,
    )


class StreamingPnLCalibEstimator:
    def __init__(
        self,
        runtime: PnLCalibRuntime,
        geometry: Optional[PitchGeometry] = None,
        keypoint_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        pnl_refine: bool = True,
        pixels_per_meter: int = 8,
        temporal_blend: float = 0.20,
    ) -> None:
        self.runtime = runtime
        self.geometry = PitchGeometry() if geometry is None else geometry
        self.keypoint_threshold = keypoint_threshold
        self.line_threshold = line_threshold
        self.pnl_refine = pnl_refine
        self.pixels_per_meter = pixels_per_meter
        self.temporal_blend = temporal_blend
        self.previous_homography_image_to_field: Optional[np.ndarray] = None
        self.last_timing_detail: Dict[str, float] = {}

    def estimate(self, frame_bgr: np.ndarray) -> PnLCalibEstimate:
        streaming_start = perf_counter()
        estimate = estimate_pnlcalib_frame(
            frame_bgr,
            runtime=self.runtime,
            geometry=self.geometry,
            keypoint_threshold=self.keypoint_threshold,
            line_threshold=self.line_threshold,
            pnl_refine=self.pnl_refine,
            pixels_per_meter=self.pixels_per_meter,
        )

        base_timing = dict(estimate.timing_detail or {})
        current_homography = estimate.homography_image_to_field
        temporal_smoothing_s = 0.0
        previous_fallback_s = 0.0
        temporal_smoothed = 0.0
        previous_fallback_applied = 0.0

        if current_homography is None and self.previous_homography_image_to_field is not None:
            previous_fallback_start = perf_counter()
            fallback_homography = self.previous_homography_image_to_field.copy()
            fallback_template = _to_template_homography(
                fallback_homography.copy(),
                pixels_per_meter=self.pixels_per_meter,
            )
            previous_fallback_s = perf_counter() - previous_fallback_start
            previous_fallback_applied = 1.0
            timing_detail = {
                **base_timing,
                "pnl_previous_fallback_s": float(previous_fallback_s),
                "pnl_temporal_smoothing_s": float(temporal_smoothing_s),
                "pnl_previous_fallback_applied": float(previous_fallback_applied),
                "pnl_temporal_smoothed": float(temporal_smoothed),
                "pnl_streaming_total_s": float(perf_counter() - streaming_start),
            }
            self.last_timing_detail = timing_detail
            return replace(
                estimate,
                homography_image_to_field=fallback_homography,
                homography_image_to_template=fallback_template,
                estimation_mode=f"{estimate.estimation_mode}+previous_fallback",
                timing_detail=timing_detail,
            )

        if current_homography is not None and self.previous_homography_image_to_field is not None:
            temporal_smoothing_start = perf_counter()
            previous_h = self.previous_homography_image_to_field.copy()
            current_h = current_homography.copy()
            previous_h /= previous_h[2, 2]
            current_h /= current_h[2, 2]
            blended = self.temporal_blend * previous_h + (1.0 - self.temporal_blend) * current_h
            blended /= blended[2, 2]
            self.previous_homography_image_to_field = blended
            temporal_template = _to_template_homography(
                blended,
                pixels_per_meter=self.pixels_per_meter,
            )
            temporal_smoothing_s = perf_counter() - temporal_smoothing_start
            temporal_smoothed = 1.0
            timing_detail = {
                **base_timing,
                "pnl_previous_fallback_s": float(previous_fallback_s),
                "pnl_temporal_smoothing_s": float(temporal_smoothing_s),
                "pnl_previous_fallback_applied": float(previous_fallback_applied),
                "pnl_temporal_smoothed": float(temporal_smoothed),
                "pnl_streaming_total_s": float(perf_counter() - streaming_start),
            }
            self.last_timing_detail = timing_detail
            return replace(
                estimate,
                homography_image_to_field=blended,
                homography_image_to_template=temporal_template,
                estimation_mode=f"{estimate.estimation_mode}+smoothed",
                timing_detail=timing_detail,
            )

        if current_homography is not None:
            self.previous_homography_image_to_field = current_homography.copy()
        timing_detail = {
            **base_timing,
            "pnl_previous_fallback_s": float(previous_fallback_s),
            "pnl_temporal_smoothing_s": float(temporal_smoothing_s),
            "pnl_previous_fallback_applied": float(previous_fallback_applied),
            "pnl_temporal_smoothed": float(temporal_smoothed),
            "pnl_streaming_total_s": float(perf_counter() - streaming_start),
        }
        self.last_timing_detail = timing_detail
        return replace(estimate, timing_detail=timing_detail)


def overlay_pnlcalib_detections(
    frame_bgr: np.ndarray,
    estimate: PnLCalibEstimate,
    draw_labels: bool = True,
) -> np.ndarray:
    output = frame_bgr.copy()
    for key, point in estimate.keypoints_dict.items():
        x_coord = int(round(point["x"]))
        y_coord = int(round(point["y"]))
        cv2.circle(output, (x_coord, y_coord), 4, (0, 255, 0), -1)
        if draw_labels:
            cv2.putText(
                output,
                str(key),
                (x_coord + 3, y_coord - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    for key, line in estimate.lines_dict.items():
        p1 = (int(round(line["x_1"])), int(round(line["y_1"])))
        p2 = (int(round(line["x_2"])), int(round(line["y_2"])))
        cv2.line(output, p1, p2, (40, 220, 255), 2)
        if draw_labels:
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                output,
                f"L{key}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"L{key}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return output


def warp_frame_to_birdeye(
    frame_bgr: np.ndarray,
    homography_image_to_template: np.ndarray,
    geometry: Optional[PitchGeometry] = None,
    pixels_per_meter: int = 8,
) -> np.ndarray:
    if geometry is None:
        geometry = PitchGeometry()
    width_px = int(round(geometry.field_length_m * pixels_per_meter))
    height_px = int(round(geometry.field_width_m * pixels_per_meter))
    return cv2.warpPerspective(
        frame_bgr,
        homography_image_to_template,
        (width_px, height_px),
        flags=cv2.INTER_LINEAR,
    )


def blend_birdeye_with_template(
    warped_frame_bgr: np.ndarray,
    geometry: Optional[PitchGeometry] = None,
    pixels_per_meter: int = 8,
    frame_alpha: float = 0.55,
) -> np.ndarray:
    if geometry is None:
        geometry = PitchGeometry()
    template_rgb, _, _, _, _ = build_pitch_template(
        field_length_m=geometry.field_length_m,
        field_width_m=geometry.field_width_m,
        pixels_per_meter=pixels_per_meter,
    )
    warped_rgb = cv2.cvtColor(warped_frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(warped_rgb, frame_alpha, template_rgb, 1.0 - frame_alpha, 0.0)


def extract_track_ground_points(
    tracks: Dict[str, object],
    frame_index: int,
    classes: Sequence[str] = ("player", "goalkeeper", "referee"),
    bottom_offset_ratio: float = 0.04,
) -> list[Dict[str, object]]:
    projected_points: list[Dict[str, object]] = []
    for class_name in classes:
        class_frames = tracks.get(class_name, [])
        if frame_index >= len(class_frames):
            continue
        frame_tracks = class_frames[frame_index]
        for track_id, track_data in frame_tracks.items():
            bbox = track_data.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in bbox]
            bbox_height = max(y2 - y1, 1.0)
            x_coord = 0.5 * (x1 + x2)
            y_coord = y2 - (bottom_offset_ratio * bbox_height)
            projected_points.append(
                {
                    "track_id": str(track_id),
                    "class_name": class_name,
                    "team": track_data.get("team"),
                    "image_point": (x_coord, y_coord),
                    "bbox": [x1, y1, x2, y2],
                    "ground_point_offset_ratio": float(bottom_offset_ratio),
                }
            )
    return projected_points


def rescale_track_points(
    track_points: Sequence[Dict[str, object]],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> list[Dict[str, object]]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_size debe tener dimensiones positivas.")

    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)

    scaled_points: list[Dict[str, object]] = []
    for track in track_points:
        image_x, image_y = track["image_point"]
        scaled_track = dict(track)
        scaled_track["image_point_original"] = (float(image_x), float(image_y))
        scaled_track["image_point"] = (
            float(image_x) * scale_x,
            float(image_y) * scale_y,
        )
        bbox = track.get("bbox")
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            scaled_track["bbox_original"] = [float(x1), float(y1), float(x2), float(y2)]
            scaled_track["bbox"] = [
                float(x1) * scale_x,
                float(y1) * scale_y,
                float(x2) * scale_x,
                float(y2) * scale_y,
            ]
        scaled_points.append(scaled_track)
    return scaled_points


def render_projected_tracks_birdeye(
    field_points_m: np.ndarray,
    teams: Optional[Sequence[Optional[str]]] = None,
    geometry: Optional[PitchGeometry] = None,
    pixels_per_meter: int = 8,
    background_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    if geometry is None:
        geometry = PitchGeometry()
    if background_rgb is None:
        template_rgb, _, _, _, _ = build_pitch_template(
            field_length_m=geometry.field_length_m,
            field_width_m=geometry.field_width_m,
            pixels_per_meter=pixels_per_meter,
        )
    else:
        template_rgb = np.asarray(background_rgb, dtype=np.uint8).copy()

    team_colors = {
        "Real Madrid": (230, 230, 230),
        "Wolfsburgo": (60, 240, 120),
        None: (240, 180, 50),
    }
    points_array = np.asarray(field_points_m, dtype=np.float32)
    if teams is None:
        teams = [None] * len(points_array)

    for (x_coord, y_coord), team_name in zip(points_array, teams):
        px = int(round(x_coord * pixels_per_meter))
        py = int(round(y_coord * pixels_per_meter))
        if 0 <= px < template_rgb.shape[1] and 0 <= py < template_rgb.shape[0]:
            color = team_colors.get(team_name, (50, 120, 240))
            cv2.circle(template_rgb, (px, py), 6, color, -1)
            cv2.circle(template_rgb, (px, py), 8, (0, 0, 0), 1)
    return template_rgb


def project_image_points(points_xy: np.ndarray, homography_image_to_field: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_xy, homography_image_to_field)
    return transformed.reshape(-1, 2)


def points_inside_field_mask(
    field_points_m: np.ndarray,
    geometry: Optional[PitchGeometry] = None,
    margin_m: float = 2.0,
) -> np.ndarray:
    if geometry is None:
        geometry = PitchGeometry()
    points_array = np.asarray(field_points_m, dtype=np.float32)
    x_values = points_array[:, 0]
    y_values = points_array[:, 1]
    return (
        (x_values >= -margin_m)
        & (x_values <= geometry.field_length_m + margin_m)
        & (y_values >= -margin_m)
        & (y_values <= geometry.field_width_m + margin_m)
    )


def save_sequence_summary(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(list(rows), file_handle, ensure_ascii=False, indent=2)
