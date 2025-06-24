import torch
import numpy as np
from scipy import ndimage
import cv2


def connected_component_loss(pred_mask, edge_distance=50, min_area=1000, penalty_weight=0.1):
    """
        计算分割结果的连通域惩罚损失，抑制小连通域和边缘连通域

        Args:
            pred_mask: 模型预测的掩码，形状为[B, H, W]，值为0-1的概率
            edge_distance: 边缘距离阈值，距离边缘小于此值的连通域视为边缘连通域
            min_area: 最小面积阈值，面积小于此值的连通域视为小连通域
            penalty_weight: 惩罚项权重，控制惩罚强度

        Returns:
            penalty_loss: 连通域惩罚损失值
    """
    batch_size = pred_mask.size(0)
    penalty_loss = 0.0

    for i in range(batch_size):
        # 将预测掩码转换为二值图
        binary_mask = (pred_mask[i] > 0.5).cpu().numpy().astype(np.uint8)

        # 查找连通域（OpenCV方法）
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = binary_mask.shape
        for contour in contours:
            # 计算连通域面积
            area = cv2.contourArea(contour)

            if area < min_area:
                # 小连通域惩罚
                area_penalty = 1.0 - (area / min_area)
                penalty_loss += area_penalty
                continue

            # 计算连通域边界框
            x, y, w_contour, h_contour = cv2.boundingRect(contour)
            center_x = x + w_contour // 2
            center_y = y + h_contour // 2

            # 计算到边缘的距离
            distance_to_edge = min(
                center_x, w - center_x, center_y, h - center_y
            )

            if distance_to_edge < edge_distance:
                # 边缘连通域惩罚
                edge_penalty = 1.0 - (distance_to_edge / edge_distance)
                penalty_loss += edge_penalty

    # 归一化损失
    penalty_loss = penalty_loss / batch_size * penalty_weight
    return penalty_loss