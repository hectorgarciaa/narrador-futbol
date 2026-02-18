"""
Módulo de configuración para el sistema de narración de fútbol con IA.

Este módulo proporciona una clase Config para cargar y acceder a la configuración
del proyecto desde un archivo YAML, con soporte para rutas absolutas y validación.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class Config:
    """
    Clase para gestionar la configuración del proyecto.
    
    Carga configuración desde config.yaml y proporciona acceso a todos los parámetros
    del sistema, con resolución de rutas relativas a absolutas.
    """
    
    def __init__(self, config_dict: Dict[str, Any], project_root: Path):
        """
        Inicializa la configuración.
        
        Args:
            config_dict: Diccionario con la configuración cargada
            project_root: Ruta raíz del proyecto
        """
        self._config = config_dict
        self.project_root = project_root
        
    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "Config":
        """
        Carga la configuración desde un archivo YAML.
        
        Args:
            config_path: Ruta al archivo de configuración. Si es None, busca
                        config.yaml en la raíz del proyecto.
                        
        Returns:
            Instancia de Config con la configuración cargada
            
        Raises:
            FileNotFoundError: Si el archivo de configuración no existe
            yaml.YAMLError: Si hay un error al parsear el YAML
        """
        # Determinar la raíz del proyecto (donde está config.yaml)
        if config_path is None:
            # Buscar config.yaml desde el archivo actual hacia arriba
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # football_ai/core -> football_ai -> narrador-futbol
            config_path = project_root / "config.yaml"
        else:
            config_path = Path(config_path).resolve()
            project_root = config_path.parent
            
        if not config_path.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado: {config_path}\n"
                f"Asegúrate de que config.yaml existe en la raíz del proyecto."
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error al parsear {config_path}: {e}")
        
        return cls(config_dict, project_root)
    
    def get_path(self, *keys: str, create_if_missing: bool = False) -> Path:
        """
        Obtiene una ruta absoluta desde la configuración.
        
        Args:
            *keys: Claves para navegar en el diccionario de configuración
                  Ejemplo: 'paths', 'models', 'yolo_base'
            create_if_missing: Si True, crea el directorio si no existe
            
        Returns:
            Path absoluto resuelto desde la raíz del proyecto
            
        Raises:
            KeyError: Si la clave no existe en la configuración
        """
        value = self._get_nested(*keys)
        
        if not isinstance(value, str):
            raise ValueError(
                f"El valor en {'.'.join(keys)} no es una ruta válida: {value}"
            )
        
        # Convertir ruta relativa a absoluta
        abs_path = self.project_root / value
        
        # Crear directorio si se solicita
        if create_if_missing and not abs_path.exists():
            if '.' in abs_path.name:  # Es un archivo
                abs_path.parent.mkdir(parents=True, exist_ok=True)
            else:  # Es un directorio
                abs_path.mkdir(parents=True, exist_ok=True)
        
        return abs_path
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Obtiene un valor de la configuración.
        
        Args:
            *keys: Claves para navegar en el diccionario
            default: Valor por defecto si la clave no existe
            
        Returns:
            Valor de la configuración o default si no existe
        """
        try:
            return self._get_nested(*keys)
        except KeyError:
            return default
    
    def _get_nested(self, *keys: str) -> Any:
        """
        Navega por el diccionario de configuración usando claves anidadas.
        
        Args:
            *keys: Secuencia de claves para acceder al valor
            
        Returns:
            Valor encontrado en la configuración
            
        Raises:
            KeyError: Si alguna clave no existe
        """
        value = self._config
        for key in keys:
            if not isinstance(value, dict):
                raise KeyError(
                    f"No se puede acceder a '{key}' en {'.'.join(keys[:-1])}"
                )
            if key not in value:
                raise KeyError(
                    f"Clave '{key}' no encontrada en configuración. "
                    f"Ruta: {'.'.join(keys)}"
                )
            value = value[key]
        return value
    
    @property
    def paths(self) -> Dict[str, Any]:
        """Diccionario con todas las rutas configuradas."""
        return self._config.get('paths', {})
    
    @property
    def detection(self) -> Dict[str, Any]:
        """Configuración de detección."""
        return self._config.get('detection', {})
    
    @property
    def tracking(self) -> Dict[str, Any]:
        """Configuración de tracking."""
        return self._config.get('tracking', {})
    
    @property
    def teams(self) -> Dict[str, Any]:
        """Configuración de equipos."""
        return self._config.get('teams', {})
    
    @property
    def visualization(self) -> Dict[str, Any]:
        """Configuración de visualización."""
        return self._config.get('visualization', {})
    
    @property
    def finetuning(self) -> Dict[str, Any]:
        """Configuración de fine-tuning."""
        return self._config.get('finetuning', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Configuración de logging."""
        return self._config.get('logging', {})
    
    def get_team_colors(self) -> Dict[str, np.ndarray]:
        """
        Obtiene los colores de los equipos como arrays de numpy.
        
        Returns:
            Diccionario con nombre del equipo como clave y color RGB como array numpy
        """
        teams = self.teams
        team_colors = {}
        
        for team_name, team_info in teams.items():
            if 'color_rgb' in team_info:
                team_colors[team_name] = np.array(team_info['color_rgb'])
        
        return team_colors
    
    def get_visualization_colors(self) -> Dict[str, tuple]:
        """
        Obtiene los colores de visualización como tuplas (para OpenCV).
        
        Returns:
            Diccionario con nombre de clase como clave y color BGR como tupla
        """
        colors = self.visualization.get('colors', {})
        return {k: tuple(v) for k, v in colors.items()}
    
    def __repr__(self) -> str:
        """Representación en string de la configuración."""
        return f"Config(project_root={self.project_root})"


# Instancia global de configuración (se carga bajo demanda)
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Obtiene la configuración global del proyecto.
    
    Esta función implementa un patrón singleton para la configuración,
    cargándola solo una vez y reutilizándola en llamadas posteriores.
    
    Args:
        config_path: Ruta al archivo de configuración (solo usado en la primera llamada)
        
    Returns:
        Instancia global de Config
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config.from_yaml(config_path)
    
    return _global_config
