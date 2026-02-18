"""
Utilidad de serialización para convertir objetos numpy a tipos nativos de Python.
"""

import numpy as np


def convert_to_serializable(obj):
    """Convierte objetos numpy a tipos serializables en JSON."""
    if isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj
