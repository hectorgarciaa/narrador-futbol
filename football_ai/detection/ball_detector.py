"""
Ball detector with reduced reg_max.

DetectR8 inherits from ultralytics Detect and reduces reg_max from 16 (default in YOLO)
to 8. This decreases the granularity of bounding box regression, which
can be beneficial for small objects like the ball, as it reduces
model complexity and the risk of overfitting in location
predictions.
"""

import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL


class DetectR8(Detect):
    """
    YOLO detection head with reg_max=8 instead of the default 16.
    
    Reduces the granularity of DFL (Distribution Focal Loss) for bounding
    box regression, useful for detecting small objects like the ball.
    
    Args:
        nc: Number of classes to detect
        ch: Tuple with input channels from each FPN scale
    """
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        old_ch = ch

        self.reg_max = 8
        self.no = nc + self.reg_max * 4

        self._build_heads(old_ch)

    def _build_heads(self, ch):
        """Rebuild the regression heads with the new reg_max."""
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
