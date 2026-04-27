from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def serialize_for_trace(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): serialize_for_trace(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [serialize_for_trace(item) for item in value]
    if isinstance(value, list):
        return [serialize_for_trace(item) for item in value]
    return value
