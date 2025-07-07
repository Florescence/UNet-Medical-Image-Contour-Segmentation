import torch
import torch.nn.functional as F


def boundary_loss(pred_mask, target_mask, edge_width=64, edge_weight=5.0, smooth=1e-6):
    """
    计算边界损失，支持多通道预测掩码 [B, C, H, W] 和3维目标掩码 [B, H, W]

    Args:
        pred_mask: 预测掩码 [B, C, H, W] 或 [B, H, W]
        target_mask: 目标掩码 [B, H, W]，支持多类别编码（如0=背景, 128=初始, 255=前景）
        edge_width: 边缘区域宽度（像素）
        edge_weight: 边缘区域惩罚权重
        smooth: 平滑因子，防止除零错误

    Returns:
        loss: 加权边界损失
    """
    # 处理4维预测掩码 [B, C, H, W]
    if pred_mask.dim() == 4:
        # 对于多通道预测，假设为类别概率分布，取前景通道
        if pred_mask.size(1) > 1:
            pred_mask = pred_mask[:, 1, :, :]  # 提取前景通道 [B, H, W]
        else:
            pred_mask = pred_mask.squeeze(1)  # 单通道直接压缩

    # 检查是否需要应用sigmoid（确保数值稳定性）
    if pred_mask.min() < -10 or pred_mask.max() > 10:  # 大数值表示可能是logits
        pred_mask = torch.sigmoid(pred_mask)

    batch_size, height, width = pred_mask.shape

    # 1. 生成边缘区域掩码
    edge_mask = _generate_edge_mask(batch_size, height, width, edge_width, device=pred_mask.device)

    # 2. 二值化目标掩码（假设255为前景）
    binary_target = (target_mask == 255).float()  # 新编码：255表示前景目标

    # 3. 计算普通区域与边缘区域损失
    normal_loss = _compute_regular_loss(pred_mask, binary_target, ~edge_mask, smooth)
    edge_loss = _compute_regular_loss(pred_mask, binary_target, edge_mask, smooth)

    # 4. 加权组合损失
    total_loss = (normal_loss + edge_weight * edge_loss) / (1 + edge_weight)
    return total_loss


def _generate_edge_mask(batch_size, height, width, edge_width, device):
    """生成边缘区域掩码（上、下、左、右边缘）"""
    edge_mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=device)
    if edge_width == 0:
        return edge_mask

    # 四周边缘区域标记为True
    edge_mask[:, :edge_width, :] = True  # 上边缘
    edge_mask[:, -edge_width:, :] = True  # 下边缘
    edge_mask[:, :, :edge_width] = True  # 左边缘
    edge_mask[:, :, -edge_width:] = True  # 右边缘
    return edge_mask


def _compute_regular_loss(pred, target, region_mask, smooth):
    """计算指定区域的边界损失（基于改进的IoU + BCEWithLogits）"""
    if not region_mask.any():
        return torch.tensor(0.0, device=pred.device)

    # 提取区域内的预测和目标
    pred_region = pred[region_mask]
    target_region = target[region_mask].float()

    # 调整为2D张量 [B*N, 1, H', W']
    b_size = pred.size(0)
    n_pixels = pred_region.numel() // b_size
    pred_2d = pred_region.view(b_size, 1, n_pixels, 1)
    target_2d = target_region.view(b_size, 1, n_pixels, 1)

    # 计算边界（使用形态学操作）
    pred_boundary = _extract_boundary(pred_2d)
    target_boundary = _extract_boundary(target_2d)

    # 展平计算IoU
    pred_flat = pred_boundary.view(-1)
    target_flat = target_boundary.view(-1)

    # 计算改进的边界损失
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    # 使用BCEWithLogits替代BCELoss，确保数值稳定性和混合精度兼容性
    # 先将预测值转换为logits（通过logit函数）
    pred_logits = _sigmoid_to_logits(pred_flat.clamp(1e-6, 1 - 1e-6))
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_flat, reduction='sum') / pred_flat.size(0)

    return (1 - iou) + 0.5 * bce


def _extract_boundary(mask, kernel_size=3):
    """使用形态学操作提取边界（支持浮点概率图）"""
    # 二值化概率图
    binary_mask = (mask > 0.5).float()

    # 创建卷积核
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)

    # 膨胀和腐蚀操作
    dilated = F.conv2d(binary_mask, kernel, padding=kernel_size // 2) > 0
    eroded = F.conv2d(binary_mask, kernel, padding=kernel_size // 2) == kernel_size ** 2

    # 边界为膨胀与腐蚀的差异
    boundary = (dilated != eroded).float()
    return boundary


def _sigmoid_to_logits(p, eps=1e-12):
    """将sigmoid输出转换回logits（用于BCEWithLogits）"""
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))