from __future__ import annotations

import importlib
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.transforms as T
import yaml

from .common import find_project_root


@dataclass
class PnLCalibRuntime:
    repo_dir: Path
    weights_kp_path: Path
    weights_line_path: Path
    device: str
    model_kp: Any
    model_line: Any
    resize_transform: Any
    framebyframe_calib_cls: Any
    get_keypoints_from_heatmap_batch_maxpool: Any
    get_keypoints_from_heatmap_batch_maxpool_l: Any
    complete_keypoints: Any
    coords_to_dict: Any


PNLCALIB_REPO_URL = "https://github.com/mguti97/PnLCalib"
PNLCALIB_REPO_DIRNAME = "pnlcalib"
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


def load_pnlcalib_runtime(device: Optional[str] = None) -> PnLCalibRuntime:
    missing_modules = _get_missing_pnlcalib_modules()
    if missing_modules:
        missing_description = ", ".join(
            f"{module} -> pip install {package}"
            for module, package in sorted(missing_modules.items())
        )
        raise ImportError(
            f"Faltan dependencias para PnLCalib: {missing_description}"
        )

    resolved_repo_dir = _ensure_pnlcalib_repo()
    resolved_kp_path, resolved_line_path = _ensure_pnlcalib_weights()
    imported = _import_pnlcalib_modules(resolved_repo_dir)

    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = resolved_repo_dir / "config" / "hrnetv2_w48.yaml"
    cfg_l_path = resolved_repo_dir / "config" / "hrnetv2_w48_l.yaml"

    with open(cfg_path, "r", encoding="utf-8") as file_handle:
        cfg = yaml.safe_load(file_handle)
    with open(cfg_l_path, "r", encoding="utf-8") as file_handle:
        cfg_l = yaml.safe_load(file_handle)

    loaded_state_kp = torch.load(resolved_kp_path, map_location=resolved_device, weights_only=False)
    loaded_state_line = torch.load(resolved_line_path, map_location=resolved_device, weights_only=False)

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

def _ensure_pnlcalib_repo(project_root: Optional[Path] = None, repo_url: str = PNLCALIB_REPO_URL) -> Path:
    target_dir = _get_default_pnlcalib_repo_path(project_root)
    if target_dir.exists():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
        check=True,
    )
    return target_dir

def _ensure_pnlcalib_weights(project_root: Optional[Path] = None) -> Tuple[Path, Path]:
    kp_path, line_path = _get_default_pnlcalib_weight_paths(project_root)

    kp_path.parent.mkdir(parents=True, exist_ok=True)
    line_path.parent.mkdir(parents=True, exist_ok=True)

    if not kp_path.exists():
        urllib.request.urlretrieve(PNLCALIB_WEIGHTS["SV_kp"], kp_path)
    if not line_path.exists():
        urllib.request.urlretrieve(PNLCALIB_WEIGHTS["SV_lines"], line_path)

    return kp_path, line_path

def _get_default_pnlcalib_repo_path(project_root: Optional[Path] = None) -> Path:
    root = find_project_root(project_root)
    return root / "external" / PNLCALIB_REPO_DIRNAME

def _get_default_pnlcalib_weights_dir(project_root: Optional[Path] = None) -> Path:
    root = find_project_root(project_root)
    return root / "models" / "pnlcalib"

def _get_default_pnlcalib_weight_paths(project_root: Optional[Path] = None) -> Tuple[Path, Path]:
    weights_dir = _get_default_pnlcalib_weights_dir(project_root)
    return weights_dir / "SV_kp", weights_dir / "SV_lines"

def _get_missing_pnlcalib_modules() -> Dict[str, str]:
    missing: Dict[str, str] = {}
    for module_name, package_name in PNLCALIB_REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing[module_name] = package_name
    return missing

def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

def _clear_conflicting_modules(repo_dir: Path, module_roots: tuple[str, ...]) -> None:
    resolved_repo_dir = repo_dir.resolve()
    for module_name in list(sys.modules.keys()):
        if not any(
            module_name == module_root or module_name.startswith(f"{module_root}.")
            for module_root in module_roots
        ):
            continue
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(module_name, None)
            continue
        try:
            resolved_module_file = Path(module_file).resolve()
        except Exception:
            sys.modules.pop(module_name, None)
            continue
        if resolved_repo_dir not in resolved_module_file.parents:
            sys.modules.pop(module_name, None)

def _register_repo_package(repo_dir: Path, package_name: str) -> None:
    package_dir = (repo_dir / package_name).resolve()
    package_module = ModuleType(package_name)
    package_module.__file__ = str(package_dir)
    package_module.__path__ = [str(package_dir)]
    package_module.__package__ = package_name
    sys.modules[package_name] = package_module

def _import_pnlcalib_modules(repo_dir: Path) -> Dict[str, Any]:
    _prepend_sys_path(repo_dir)
    _clear_conflicting_modules(repo_dir, ("utils", "model"))
    _register_repo_package(repo_dir, "utils")
    _register_repo_package(repo_dir, "model")
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

__all__ = [ "load_pnlcalib_runtime" ]
