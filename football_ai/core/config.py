"""
Configuration module for the AI football commentary system.

This module provides a Config class to load and access project configuration
from a YAML file, with support for absolute paths and validation.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


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
    def team_detector(self) -> Dict[str, Any]:
        """Team Detector configuration."""
        return self._config.get('team_detector', {})
    
    @property
    def tracking(self) -> Dict[str, Any]:
        """Tracking configuration."""
        return self._config.get('tracking', {})
    
    @property
    def projector(self) -> Dict[str, Any]:
        """Projector configuration."""
        return self._config.get('projector', {})
    
    @property
    def bytetracker(self) -> Dict[str, Any]:
        """ByteTracker configuration."""
        return self._config.get('bytetracker', {})
    
    @property
    def posession(self) -> Dict[str, Any]:
        """Possession configuration."""
        return self._config.get('posession', {})

    @property
    def actions(self) -> Dict[str, Any]:
        """Actions configuration."""
        return self._config.get('actions', {})

    @property
    def commentary(self) -> Dict[str, Any]:
        """Commentary configuration."""
        return self._config.get('commentary', {})

    @property
    def canonical(self) -> Dict[str, Any]:
        """Canonical tracking configuration."""
        return self._config.get('canonical', {})

    @property
    def positions(self) -> Dict[str, Any]:
        """Position inference and stabilization configuration."""
        return self._config.get('positions', {})
    
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
    def logging_config(self) -> Dict[str, Any]:
        """Logging configuration."""
        return self._config.get('logging', {})
    
    def get_team_names(self) -> list:
        """
        Get the list of configured team names.
        
        Returns:
            List with team names
        """
        return list(self.teams.keys())
    
    def get_team_colors(self) -> Dict[str, np.ndarray]:
        """
        Get team colors as numpy arrays.
        
        Returns:
            Dictionary with team name as key and color reference as numpy array.
            Supports both `color_lab_opencv` (preferred) and legacy `color_rgb`.
        """
        teams = self.teams
        team_colors = {}
        
        for team_name, team_info in teams.items():
            if 'color_lab_opencv' in team_info:
                team_colors[team_name] = np.array(
                    team_info['color_lab_opencv'], dtype=np.float32
                )
            elif 'color_rgb' in team_info:
                team_colors[team_name] = np.array(team_info['color_rgb'], dtype=np.float32)
        
        return team_colors
    
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
