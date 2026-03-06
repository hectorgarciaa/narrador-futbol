"""
Configuration module for the AI football commentary system.

This module provides a Config class to load and access project configuration
from a YAML file, with support for absolute paths and validation.
"""

import yaml
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2


class Config:
    """
    Class to manage project configuration.
    
    Loads configuration from config.yaml and provides access to all system
    parameters, with relative-to-absolute path resolution.
    """
    
    def __init__(self, config_dict: Dict[str, Any], project_root: Path):
        """
        Initialize the configuration.
        
        Args:
            config_dict: Dictionary with the loaded configuration
            project_root: Project root path
        """
        self._config = config_dict
        self.project_root = project_root
        
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file. If None, searches for
                        config.yaml in the project root.
                        
        Returns:
            Config instance with the loaded configuration
            
        Raises:
            FileNotFoundError: If the configuration file does not exist
            yaml.YAMLError: If there is an error parsing the YAML
        """
        # Determine the project root (where config.yaml is)
        if config_path is None:
            # Search for config.yaml from the current file upward
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # football_ai/core -> football_ai -> narrador-futbol
            config_path = project_root / "config.yaml"
        else:
            config_path = Path(config_path).resolve()
            project_root = config_path.parent
            
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Make sure config.yaml exists in the project root."
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing {config_path}: {e}")
        
        return cls(config_dict, project_root)
    
    def get_path(self, *keys: str, create_if_missing: bool = False) -> Path:
        """
        Get an absolute path from the configuration.
        
        Args:
            *keys: Keys to navigate the configuration dictionary
                  Example: 'paths', 'models', 'yolo_base'
            create_if_missing: If True, create the directory if it doesn't exist
            
        Returns:
            Absolute Path resolved from the project root
            
        Raises:
            KeyError: If the key does not exist in the configuration
        """
        value = self._get_nested(*keys)
        
        if not isinstance(value, str):
            raise ValueError(
                f"The value at {'.'.join(keys)} is not a valid path: {value}"
            )
        
        # Convert relative path to absolute
        abs_path = self.project_root / value
        
        # Create directory if requested
        if create_if_missing and not abs_path.exists():
            if '.' in abs_path.name:  # It's a file
                abs_path.parent.mkdir(parents=True, exist_ok=True)
            else:  # It's a directory
                abs_path.mkdir(parents=True, exist_ok=True)
        
        return abs_path
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a value from the configuration.
        
        Args:
            *keys: Keys to navigate the dictionary
            default: Default value if the key does not exist
            
        Returns:
            Configuration value or default if not found
        """
        try:
            return self._get_nested(*keys)
        except KeyError:
            return default
    
    def _get_nested(self, *keys: str) -> Any:
        """
        Navigate the configuration dictionary using nested keys.
        
        Args:
            *keys: Sequence of keys to access the value
            
        Returns:
            Value found in the configuration
            
        Raises:
            KeyError: If any key does not exist
        """
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                raise KeyError(
                    f"Cannot access '{key}' in {'.'.join(keys[:-1])}"
                )
            if key not in value:
                raise KeyError(
                    f"Key '{key}' not found in configuration. "
                    f"Path: {'.'.join(keys)}"
                )
            value = value[key]
        return value
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Dictionary with all configured paths."""
        return self._config.get('paths', {})
    
    @property
    def detection(self) -> Dict[str, Any]:
        """Detection configuration."""
        return self._config.get('detection', {})
    
    @property
    def tracking(self) -> Dict[str, Any]:
        """Tracking configuration."""
        return self._config.get('tracking', {})
    
    @property
    def teams(self) -> Dict[str, Any]:
        """Teams configuration."""
        return self._config.get('teams', {})
    
    @property
    def visualization(self) -> Dict[str, Any]:
        """Visualization configuration."""
        return self._config.get('visualization', {})
    
    @property
    def finetuning(self) -> Dict[str, Any]:
        """Fine-tuning configuration."""
        return self._config.get('finetuning', {})
    
    @property
    def color_clustering(self) -> Dict[str, Any]:
        """Color clustering configuration."""
        return self._config.get('color_clustering', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Logging configuration."""
        return self._config.get('logging', {})
    
    def get_team_names(self) -> list:
        """
        Get the list of configured team names.
        
        Returns:
            List with team names
        """
        return list(self.get_team_colors().keys())

    @staticmethod
    def _parse_numeric_triplet(color_value: Any, field_name: str) -> np.ndarray:
        color_array = np.asarray(color_value, dtype=np.float32).reshape(-1)
        if color_array.size != 3:
            raise ValueError(
                f"'{field_name}' debe contener exactamente 3 valores. Valor recibido: {color_value}"
            )
        return color_array.astype(np.float32)

    @staticmethod
    def _parse_hex_color(color_value: str) -> np.ndarray:
        if not isinstance(color_value, str):
            raise ValueError(f"El color HEX debe ser texto, recibido: {type(color_value)}")
        hex_str = color_value.strip()
        if hex_str.startswith("#"):
            hex_str = hex_str[1:]
        if len(hex_str) == 3:
            hex_str = "".join(ch * 2 for ch in hex_str)
        if not re.fullmatch(r"[0-9a-fA-F]{6}", hex_str):
            raise ValueError(f"Color HEX inválido: {color_value}")
        return np.array(
            [int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)],
            dtype=np.float32,
        )

    @classmethod
    def _parse_rgb_like_triplet(cls, color_value: Any, field_name: str) -> np.ndarray:
        if isinstance(color_value, str):
            rgb_match = re.fullmatch(
                r"\s*rgb\s*\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)\s*",
                color_value,
                flags=re.IGNORECASE,
            )
            if rgb_match is not None:
                return np.array(
                    [float(rgb_match.group(1)), float(rgb_match.group(2)), float(rgb_match.group(3))],
                    dtype=np.float32,
                )
            if color_value.strip().startswith("#"):
                return cls._parse_hex_color(color_value)

        values = cls._parse_numeric_triplet(color_value, field_name)
        # Soporte [0..1] -> [0..255]
        if np.max(values) <= 1.0 + 1e-6 and np.min(values) >= -1e-6:
            values = values * 255.0
        values = np.clip(values, 0.0, 255.0)
        return values.astype(np.float32)

    @staticmethod
    def _rgb_to_lab(color_rgb: np.ndarray) -> np.ndarray:
        rgb_uint8 = np.clip(np.asarray(color_rgb, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        rgb_uint8 = rgb_uint8.reshape(1, 1, 3)
        color_lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).reshape(3)
        return color_lab.astype(np.float32)

    @staticmethod
    def _bgr_to_lab(color_bgr: np.ndarray) -> np.ndarray:
        bgr_uint8 = np.clip(np.asarray(color_bgr, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        bgr_uint8 = bgr_uint8.reshape(1, 1, 3)
        color_lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB).reshape(3)
        return color_lab.astype(np.float32)

    def _resolve_team_color_lab(
        self,
        team_name: str,
        team_info: Dict[str, Any],
        default_color_space: str = "lab",
    ) -> np.ndarray:
        if not isinstance(team_info, dict):
            raise ValueError(f"La configuración del equipo '{team_name}' debe ser un diccionario.")

        if "color_lab" in team_info:
            values = self._parse_numeric_triplet(team_info["color_lab"], "color_lab")
            return np.clip(values, 0.0, 255.0).astype(np.float32)

        # Compatibilidad con la configuración antigua: color_rgb se trataba ya como referencia LAB.
        if "color_rgb" in team_info and "color_space" not in team_info:
            values = self._parse_numeric_triplet(team_info["color_rgb"], "color_rgb")
            return np.clip(values, 0.0, 255.0).astype(np.float32)

        color_space = str(team_info.get("color_space", default_color_space)).strip().lower()
        if color_space in {"cie-lab", "lab"}:
            color_value = team_info.get("color", team_info.get("color_rgb", team_info.get("color_lab")))
            if color_value is None:
                raise ValueError(
                    f"El equipo '{team_name}' no tiene color definido. Usa 'color' o 'color_lab'."
                )
            values = self._parse_numeric_triplet(color_value, "color")
            return np.clip(values, 0.0, 255.0).astype(np.float32)

        if color_space == "rgb":
            color_value = team_info.get("color", team_info.get("color_rgb", team_info.get("color_hex")))
            if color_value is None:
                raise ValueError(
                    f"El equipo '{team_name}' no tiene color RGB definido."
                )
            color_rgb = self._parse_rgb_like_triplet(color_value, "color_rgb")
            return self._rgb_to_lab(color_rgb)

        if color_space == "bgr":
            color_value = team_info.get("color", team_info.get("color_bgr"))
            if color_value is None:
                raise ValueError(
                    f"El equipo '{team_name}' no tiene color BGR definido."
                )
            color_bgr = self._parse_rgb_like_triplet(color_value, "color_bgr")
            return self._bgr_to_lab(color_bgr)

        if color_space in {"hex", "html"}:
            color_value = team_info.get("color", team_info.get("color_hex"))
            if color_value is None:
                raise ValueError(
                    f"El equipo '{team_name}' no tiene color HEX definido."
                )
            color_rgb = self._parse_hex_color(str(color_value))
            return self._rgb_to_lab(color_rgb)

        raise ValueError(
            f"color_space inválido para equipo '{team_name}': {color_space}. "
            "Valores soportados: lab, rgb, bgr, hex."
        )

    def _iter_team_entries(self, team_data: Any):
        if team_data is None:
            return
        if isinstance(team_data, list):
            for item in team_data:
                if isinstance(item, dict):
                    yield None, item
            return
        if isinstance(team_data, dict):
            if "teams" in team_data:
                yield from self._iter_team_entries(team_data["teams"])
                return
            for key, value in team_data.items():
                if isinstance(value, dict):
                    yield str(key), value
            return
        raise ValueError(f"Formato de equipos no soportado: {type(team_data)}")

    def _load_teams_from_file(self, source_file: str) -> Dict[str, Any]:
        source_path = Path(source_file)
        if not source_path.is_absolute():
            source_path = self.project_root / source_path
        if not source_path.exists():
            raise FileNotFoundError(
                f"Archivo de equipos no encontrado: {source_path}"
            )

        suffix = source_path.suffix.lower()
        if suffix == ".json":
            with open(source_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if suffix in {".yaml", ".yml"}:
            with open(source_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            return loaded if loaded is not None else {}
        raise ValueError(
            f"Extensión de archivo de equipos no soportada: {source_path}. "
            "Usa .json, .yaml o .yml"
        )

    def _resolve_teams_config(self) -> Dict[str, Dict[str, Any]]:
        teams_cfg = self._config.get("teams", {})
        if not isinstance(teams_cfg, dict):
            return {}

        default_color_space = str(teams_cfg.get("default_color_space", "lab")).strip().lower()
        source_file = teams_cfg.get("source_file")
        resolved: Dict[str, Dict[str, Any]] = {}

        if source_file:
            file_cfg = self._load_teams_from_file(str(source_file))
            file_default_space = default_color_space
            if isinstance(file_cfg, dict):
                file_default_space = str(
                    file_cfg.get("default_color_space", file_default_space)
                ).strip().lower()
            for team_key, team_info in self._iter_team_entries(file_cfg):
                if not isinstance(team_info, dict):
                    continue
                team_name = str(team_info.get("name") or team_key or "").strip()
                if not team_name:
                    raise ValueError(
                        f"Entrada de equipo inválida en '{source_file}': falta campo 'name'."
                    )
                resolved[team_name] = {
                    "name": team_name,
                    "color_lab": self._resolve_team_color_lab(
                        team_name,
                        team_info,
                        default_color_space=file_default_space,
                    ),
                }

        inline_entries = {
            key: value
            for key, value in teams_cfg.items()
            if key not in {"source_file", "default_color_space"} and isinstance(value, dict)
        }
        for team_key, team_info in self._iter_team_entries(inline_entries):
            team_name = str(team_info.get("name") or team_key or "").strip()
            if not team_name:
                continue
            resolved[team_name] = {
                "name": team_name,
                "color_lab": self._resolve_team_color_lab(
                    team_name,
                    team_info,
                    default_color_space=default_color_space,
                ),
            }

        return resolved

    def get_team_colors(self) -> Dict[str, np.ndarray]:
        """
        Get team colors as numpy arrays.
        
        Returns:
            Dictionary with team name as key and LAB reference color as numpy array
        """
        resolved_teams = self._resolve_teams_config()
        return {
            team_name: np.asarray(team_info["color_lab"], dtype=np.float32)
            for team_name, team_info in resolved_teams.items()
        }
    
    def get_visualization_colors(self) -> Dict[str, tuple]:
        """
        Get visualization colors as tuples (for OpenCV).
        
        Returns:
            Dictionary with class name as key and BGR color as tuple
        """
        colors = self.visualization.get('colors', {})
        return {k: tuple(v) for k, v in colors.items()}
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"Config(project_root={self.project_root})"


# Global configuration instance (loaded on demand)
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global project configuration.
    
    This function implements a singleton pattern for configuration,
    loading it only once and reusing it in subsequent calls.
    
    Args:
        config_path: Path to the configuration file (only used on first call)
        
    Returns:
        Global Config instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config.from_yaml(config_path)
    
    return _global_config
