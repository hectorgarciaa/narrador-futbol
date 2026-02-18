import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL

class DetectR8(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        old_ch = ch

        self.reg_max = 8
        self.no = nc + self.reg_max * 4

        # Rebuild only needed parts
        self._build_heads(old_ch)

    def _build_heads(self, ch):
        # same logic que la clase original
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
