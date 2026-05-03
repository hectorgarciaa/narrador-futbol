from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

import torch

_logger = logging.getLogger(__name__)

# Model cache — evita cargar state_dict del disco + build_model en cada checkpoint
_model_cache: dict[tuple[int, str, str], tuple[Any, tuple[Any, Any, Any]]] = {}


def _clear_model_cache() -> None:
    """Borra el cache de modelos PathCRF (util para tests o cambio de config)."""
    _model_cache.clear()


def _resolve_repo_path(repo_path: str | Path) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el repo de PathCRF: {path}")
    return path


def _ensure_pathcrf_import_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _install_torch_geometric_stub() -> None:
    if "torch_geometric" in sys.modules:
        return

    tg_module = types.ModuleType("torch_geometric")
    tg_data_module = types.ModuleType("torch_geometric.data")

    class _DummyData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _DummyBatch:
        @staticmethod
        def from_data_list(data_list: list[Any]) -> list[Any]:
            return data_list

    tg_data_module.Data = _DummyData
    tg_data_module.Batch = _DummyBatch
    tg_module.data = tg_data_module
    sys.modules["torch_geometric"] = tg_module
    sys.modules["torch_geometric.data"] = tg_data_module


def _has_real_torch_geometric() -> bool:
    try:
        importlib.import_module("torch_geometric.data")
        return True
    except ModuleNotFoundError:
        return False


def _select_device(device_name: str) -> torch.device:
    normalized = str(device_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _import_pathcrf_modules(repo_path: Path, trial_args: dict[str, Any]) -> tuple[Any, Any, Any]:
    _ensure_pathcrf_import_path(repo_path)
    agent_model = str(trial_args.get("agent_model", "")).strip().lower()
    if agent_model != "gat" and not _has_real_torch_geometric():
        _install_torch_geometric_stub()
    elif agent_model == "gat" and not _has_real_torch_geometric():
        raise ModuleNotFoundError(
            "El checkpoint de PathCRF usa `agent_model=gat`, pero `torch_geometric` no esta instalado en el entorno."
        )

    pathcrf_utils = importlib.import_module("models.utils")
    pathcrf_inference = importlib.import_module("inference")
    pathcrf_postprocess = importlib.import_module("datatools.postprocess")
    return pathcrf_utils, pathcrf_inference, pathcrf_postprocess


def get_or_load_model(
    *,
    trial: int,
    model_file: str,
    device: torch.device,
    repo_path: Path,
    save_path: Path,
    trial_args: dict[str, Any],
):
    """Carga (o recupera del cache) el modelo PathCRF y sus modulos auxiliares."""
    cache_key = (trial, model_file, str(device))
    if cache_key in _model_cache:
        _logger.debug("PathCRF model cache hit (trial=%d, device=%s)", trial, device)
        return _model_cache[cache_key]

    _logger.info("PathCRF model cache miss — loading from disk (trial=%d, device=%s)", trial, device)
    pathcrf_utils, pathcrf_inference_proc, pathcrf_postprocess_proc = _import_pathcrf_modules(repo_path, trial_args)

    model = pathcrf_utils.build_model(trial_args, device=device)
    model_path = Path(pathcrf_utils.resolve_model_path(str(save_path), model_file)).resolve()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    modules = (pathcrf_utils, pathcrf_inference_proc, pathcrf_postprocess_proc)
    _model_cache[cache_key] = (model, modules)
    return model, modules
