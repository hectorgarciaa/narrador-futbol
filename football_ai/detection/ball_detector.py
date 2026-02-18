"""
Detector de balón con reg_max reducido.

DetectR8 hereda de ultralytics Detect y reduce reg_max de 16 (por defecto en YOLO)
a 8. Esto disminuye la granularidad de la regresión de bounding boxes, lo cual
puede ser beneficioso para objetos pequeños como el balón, ya que reduce
la complejidad del modelo y el riesgo de sobreajuste en las predicciones
de localización.
"""

import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL


class DetectR8(Detect):
    """
    Cabeza de detección YOLO con reg_max=8 en lugar del default 16.
    
    Reduce la granularidad de DFL (Distribution Focal Loss) para regresión
    de bounding boxes, útil para detección de objetos pequeños como el balón.
    
    Args:
        nc: Número de clases a detectar
        ch: Tupla con canales de entrada de cada escala del FPN
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        old_ch = ch

        self.reg_max = 8
        self.no = nc + self.reg_max * 4

        self._build_heads(old_ch)

    def _build_heads(self, ch):
        """Reconstruye las cabezas de regresión con el nuevo reg_max."""
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max)
