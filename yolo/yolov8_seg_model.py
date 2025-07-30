import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.bn2(self.pw(x))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DWConv(cin, cout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='nearest')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class YOLOv8_Seg_S(nn.Module):
    """
    Ultra-light segmentation for 512×512 medical mask.
    Channels: 16-32-64-128 (all DW+PW)
    """

    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        # Encoder
        self.inc   = DWConv(n_channels, 8)
        self.down1 = DWConv(8, 16, s=2)
        self.down2 = DWConv(16, 32, s=2)
        self.down3 = DWConv(32, 64, s=2)
        self.down4 = DWConv(64, 64, s=2)

        # Decoder (通道已对齐)
        self.up1 = Up(64 + 64, 64)   # 32→64
        self.up2 = Up(64 + 32,  32)    # 64→128
        self.up3 = Up(32  + 16,  16)    # 128→256
        self.up4 = Up(16  + 8,  8)    # 256→512

        # Head
        self.seg = nn.Conv2d(8, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)        # 512×512×16
        x2 = self.down1(x1)     # 256×256×32
        x3 = self.down2(x2)     # 128×128×64
        x4 = self.down3(x3)     # 64×64×128
        x5 = self.down4(x4)     # 32×32×128

        x = self.up1(x5, x4)    # 64×64×128
        x = self.up2(x, x3)     # 128×128×64
        x = self.up3(x, x2)     # 256×256×32
        x = self.up4(x, x1)     # 512×512×16

        return self.seg(x)      # 512×512×1

    def use_checkpointing(self):
        for m in self.modules():
            if hasattr(m, 'forward'):
                m.forward = torch.utils.checkpoint.checkpoint(m.forward)