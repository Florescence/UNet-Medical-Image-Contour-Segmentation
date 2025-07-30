""" Full assembly of the parts to form the complete U-Net++ network with Spatial Attention """

import torch
import torch.nn as nn
from .unet_nested_parts import DoubleConv, Down, NestedUp, OutConv

class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, depth=4, bilinear=False, use_attention=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth
        self.base_channels = 64  # 基础通道数，与原始UNet保持一致

        # 编码器：[DoubleConv, Down, Down, ...]（depth个Down）
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(n_channels, self.base_channels))  # 0: 输入层
        for i in range(depth):
            in_ch = self.base_channels * (2 ** i)
            out_ch = self.base_channels * (2 **(i + 1))
            self.encoder.append(Down(in_ch, out_ch))  # 1~depth: 下采样层

        # 解码器：嵌套结构，共depth层解码器
        self.decoder = nn.ModuleList()
        for i in range(depth):  # 解码器层级 i（0~depth-1）
            level = nn.ModuleList()
            for j in range(depth - i):  # 每个层级的模块数
                # 计算当前解码器模块的输入通道数和跳跃连接总通道数
                # 输入通道数：来自更深层解码器的输出通道数（等于当前层的out_ch）
                # 跳跃连接总通道数：编码器对应层 + 上层解码器对应层的输出（共i+1个跳跃连接）
                in_ch = self.base_channels * (2** (depth - i))
                skip_ch = self.base_channels * (2 ** j) * (i + 1)  # i+1个跳跃连接，每个通道数为base*2^j
                out_ch = self.base_channels * (2 ** j)
                level.append(NestedUp(in_ch, skip_ch, out_ch, bilinear, use_attention))
            self.decoder.append(level)

        # 输出层：每个解码器层级对应一个输出（深度监督）
        self.outc = nn.ModuleList([OutConv(self.base_channels, n_classes) for _ in range(depth)])

    def forward(self, x):
        # 编码器输出：[x0, x1, x2, ..., xdepth]（x0是输入层，x1是第一次下采样后，以此类推）
        encoder_outs = [self.encoder[0](x)]
        for i in range(1, self.depth + 1):
            encoder_outs.append(self.encoder[i](encoder_outs[-1]))

        # 解码器输出
        decoder_outs = []
        # 第0层解码器（最浅的解码器，直接连接最深的编码器输出）
        level0 = []
        for j in range(self.depth):
            # 输入：最深的编码器输出（encoder_outs[-1]）；跳跃连接：[encoder_outs[depth - j]]
            if j == 0:
                x = self.decoder[0][j](encoder_outs[-1], [encoder_outs[-(j + 2)]])
            else:
                x = self.decoder[0][j](x, [encoder_outs[-(j + 2)]])
            level0.append(x)
        decoder_outs.append(level0)

        # 第1~depth-1层解码器
        for i in range(1, self.depth):
            leveli = []
            for j in range(self.depth - i):
                # 跳跃连接：编码器对应层 + 上层解码器的输出（共i+1个）
                skips = [encoder_outs[-(j + i + 2)]] + [decoder_outs[k][j] for k in range(i)]
                x = self.decoder[i][j](decoder_outs[i-1][j+1], skips)
                leveli.append(x)
            decoder_outs.append(leveli)

        # 深度监督：每个解码器层级的第一个输出作为该层级的预测
        preds = [self.outc[i](decoder_outs[i][0]) for i in range(self.depth)]
        return preds if self.training else preds[-1]  # 训练时返回所有预测，推理时返回最深层预测

class UNetPlusPlus_S(nn.Module):
    """Small version of U-Net++ with reduced channels"""

    def __init__(self, n_channels, n_classes, depth=4, bilinear=False, use_attention=False):
        super(UNetPlusPlus_S, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        # 基础通道数设置 (较小)
        base_channels = 16

        # 编码器部分
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(n_channels, base_channels))

        for i in range(depth):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i + 1))
            self.encoder.append(Down(in_channels, out_channels))

        # 解码器部分 (嵌套结构)
        self.decoder = nn.ModuleList()

        # 第0层解码器 (直接连接到输出)
        for i in range(depth):
            decoder_level = nn.ModuleList()
            for j in range(depth - i):
                in_channels = base_channels * (2 ** (j + i + 1))
                skip_channels = base_channels * (2 ** j) * (i + 1)
                out_channels = base_channels * (2 ** j)

                decoder_level.append(NestedUp(
                    in_channels, skip_channels, out_channels, bilinear, use_attention
                ))
            self.decoder.append(decoder_level)

        # 输出层
        self.outc = nn.ModuleList()
        for i in range(depth):
            self.outc.append(OutConv(base_channels, n_classes))

    def forward(self, x):
        # 编码器前向传播
        encoder_outputs = [self.encoder[0](x)]
        for i in range(1, self.depth + 1):
            encoder_outputs.append(self.encoder[i](encoder_outputs[i-1]))

        # 解码器前向传播
        decoder_outputs = []

        # 第0层解码器
        level_0_outputs = []
        for i in range(self.depth):
            if i == 0:
                x = self.decoder[0][i](encoder_outputs[-1], [encoder_outputs[-(i+2)]])
            else:
                x = self.decoder[0][i](x, [encoder_outputs[-(i+2)]])
            level_0_outputs.append(x)
        decoder_outputs.append(level_0_outputs)

        # 更高层解码器
        for i in range(1, self.depth):
            level_i_outputs = []
            for j in range(self.depth - i):
                skip_connections = [encoder_outputs[-(j+i+2)]]
                skip_connections += [decoder_outputs[k][j] for k in range(i)]
                x = self.decoder[i][j](decoder_outputs[i-1][j+1], skip_connections)
                level_i_outputs.append(x)
            decoder_outputs.append(level_i_outputs)

        # 输出层
        final_outputs = []
        for i in range(self.depth):
            final_outputs.append(self.outc[i](decoder_outputs[i][0]))

        # 在训练模式下返回所有输出，在推理模式下只返回最深层的输出
        if self.training:
            return final_outputs
        else:
            return final_outputs[-1]

class UNetPlusPlus_L(nn.Module):
    """Large version of U-Net++ with increased channels"""

    def __init__(self, n_channels, n_classes, depth=5, bilinear=False, use_attention=False):
        super(UNetPlusPlus_L, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.depth = depth

        # 基础通道数设置 (较大)
        base_channels = 64

        # 编码器部分
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(n_channels, base_channels))

        for i in range(depth):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i + 1))
            self.encoder.append(Down(in_channels, out_channels))

        # 解码器部分 (嵌套结构)
        self.decoder = nn.ModuleList()

        # 第0层解码器 (直接连接到输出)
        for i in range(depth):
            decoder_level = nn.ModuleList()
            for j in range(depth - i):
                in_channels = base_channels * (2 ** (j + i + 1))
                skip_channels = base_channels * (2 ** j) * (i + 1)
                out_channels = base_channels * (2 ** j)

                decoder_level.append(NestedUp(
                    in_channels, skip_channels, out_channels, bilinear, use_attention
                ))
            self.decoder.append(decoder_level)

        # 输出层
        self.outc = nn.ModuleList()
        for i in range(depth):
            self.outc.append(OutConv(base_channels, n_classes))

    def forward(self, x):
        # 编码器前向传播
        encoder_outputs = [self.encoder[0](x)]
        for i in range(1, self.depth + 1):
            encoder_outputs.append(self.encoder[i](encoder_outputs[i-1]))

        # 解码器前向传播
        decoder_outputs = []

        # 第0层解码器
        level_0_outputs = []
        for i in range(self.depth):
            if i == 0:
                x = self.decoder[0][i](encoder_outputs[-1], [encoder_outputs[-(i+2)]])
            else:
                x = self.decoder[0][i](x, [encoder_outputs[-(i+2)]])
            level_0_outputs.append(x)
        decoder_outputs.append(level_0_outputs)

        # 更高层解码器
        for i in range(1, self.depth):
            level_i_outputs = []
            for j in range(self.depth - i):
                skip_connections = [encoder_outputs[-(j+i+2)]]
                skip_connections += [decoder_outputs[k][j] for k in range(i)]
                x = self.decoder[i][j](decoder_outputs[i-1][j+1], skip_connections)
                level_i_outputs.append(x)
            decoder_outputs.append(level_i_outputs)

        # 输出层
        final_outputs = []
        for i in range(self.depth):
            final_outputs.append(self.outc[i](decoder_outputs[i][0]))

        # 在训练模式下返回所有输出，在推理模式下只返回最深层的输出
        if self.training:
            return final_outputs
        else:
            return final_outputs[-1]