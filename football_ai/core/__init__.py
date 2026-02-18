"""
Módulo core - Configuración, logging y utilidades compartidas.
"""

from .config import Config, get_config
from .logger import Logger, get_logger
from .serialization import convert_to_serializable

__all__ = ['Config', 'get_config', 'Logger', 'get_logger', 'convert_to_serializable']
