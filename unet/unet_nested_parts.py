""" Parts of the U-Net++ model with Spatial Attention """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SpatialAttention(nn.Module):
    """空间注意力模块，通过卷积生成注意力图"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的最大值和平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 合并两个特征图
        out = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积生成空间注意力图
        out = self.conv1(out)
        # 应用sigmoid激活函数
        return self.sigmoid(out)

class NestedUp(nn.Module):
    """U-Net++ 中的嵌套上采样模块，支持多跳跃连接和空间注意力"""

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.attention = SpatialAttention() if use_attention else nn.Identity()

        # 上采样层配置（核心修正：确保通道数匹配）
        if bilinear:
            # 双线性上采样：输出通道数为 out_channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 拼接后的通道数 = 上采样输出通道数 + 所有跳跃连接通道数总和
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)
        else:
            # 转置卷积上采样：输入通道=in_channels，输出通道=out_channels
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # 拼接后的通道数 = 上采样输出通道数（out_channels） + 所有跳跃连接通道数总和（skip_channels）
            self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip_connections):
        """
        x: 解码器当前层的输入特征（来自更深层的输出）
        skip_connections: 编码器对应的多层跳跃连接特征列表
        """
        # 上采样
        x = self.up(x)

        # 对每个跳跃连接应用空间注意力，并处理尺寸不匹配
        attended_skips = []
        for skip in skip_connections:
            # 应用注意力
            attended_skip = self.attention(skip) * skip
            # 对齐尺寸
            diffY = attended_skip.size()[2] - x.size()[2]
            diffX = attended_skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            attended_skips.append(attended_skip)

        # 拼接上采样特征和所有跳跃连接特征
        x = torch.cat([x] + attended_skips, dim=1)

        # 双卷积融合
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)