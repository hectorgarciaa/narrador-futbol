"""
Módulo de logging para el sistema de narración de fútbol con IA.

Proporciona configuración centralizada de logging con soporte para
múltiples niveles, formateo personalizado y salida a archivo y consola.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Clase para gestionar el logging del proyecto.
    
    Proporciona configuración centralizada de logging con salida
    a archivo y consola, niveles configurables y formateo personalizado.
    """
    
    _initialized = False
    _loggers = {}
    
    @classmethod
    def setup(
        cls,
        log_file: Optional[str] = None,
        level: str = "INFO",
        format_str: Optional[str] = None,
        log_to_console: bool = True,
        log_to_file: bool = True
    ) -> None:
        """
        Configura el sistema de logging.
        
        Args:
            log_file: Ruta al archivo de log (relativa o absoluta)
            level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_str: Formato personalizado para los mensajes de log
            log_to_console: Si True, también imprime logs en consola
            log_to_file: Si True, guarda logs en archivo
        """
        if cls._initialized:
            return
        
        # Formato por defecto
        if format_str is None:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Convertir nivel de string a constante de logging
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Crear formateador
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        
        # Configurar root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Limpiar handlers existentes
        root_logger.handlers = []
        
        # Handler para consola
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Handler para archivo
        if log_to_file and log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Obtiene un logger con el nombre especificado.
        
        Args:
            name: Nombre del logger (generalmente __name__ del módulo)
            
        Returns:
            Logger configurado
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def setup_from_config(cls, config) -> None:
        """
        Configura el logging desde un objeto Config.
        
        Args:
            config: Instancia de Config con la configuración de logging
        """
        log_config = config.logging_config
        
        log_file = None
        if 'log_file' in log_config:
            log_file = str(config.project_root / log_config['log_file'])
        
        cls.setup(
            log_file=log_file,
            level=log_config.get('level', 'INFO'),
            format_str=log_config.get('format')
        )


def get_logger(name: str) -> logging.Logger:
    """
    Función de conveniencia para obtener un logger.
    
    Args:
        name: Nombre del logger (usar __name__)
        
    Returns:
        Logger configurado
    """
    return Logger.get_logger(name)
