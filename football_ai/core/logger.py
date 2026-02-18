"""
Logging module for the AI football commentary system.

Provides centralized logging configuration with support for
multiple levels, custom formatting, and file/console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    Class to manage project logging.
    
    Provides centralized logging configuration with file and console
    output, configurable levels, and custom formatting.
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
        Configure the logging system.
        
        Args:
            log_file: Path to the log file (relative or absolute)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_str: Custom format for log messages
            log_to_console: If True, also prints logs to console
            log_to_file: If True, saves logs to file
        """
        if cls._initialized:
            return
        
        # Default format
        if format_str is None:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Convert string level to logging constant
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler
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
        Get a logger with the specified name.
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            Configured logger
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def setup_from_config(cls, config) -> None:
        """
        Configure logging from a Config object.
        
        Args:
            config: Config instance with the logging configuration
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
    Convenience function to get a logger.
    
    Args:
        name: Logger name (use __name__)
        
    Returns:
        Configured logger
    """
    return Logger.get_logger(name)
