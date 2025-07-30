import cv2
import numpy as np


def remove_internal_regions(mask, foreground_value=2, background_values=[0, 1]):
    """
    去除前景区域内部的非前景连通域（直接处理[0,1,2]格式掩码）

    Args:
        mask: 输入掩码图像，应为单通道numpy数组，值为[0,1,2]
        foreground_value: 前景像素值，默认为2
        background_values: 非前景像素值，默认为[0,1]

    Returns:
        处理后的掩码图像，值仍为[0,1,2]
    """
    # 确保输入是numpy数组o
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    # 创建掩码的副本，避免修改原始数据
    processed_mask = mask.copy()

    # 二值化：将前景设为255，非前景设为0
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask == foreground_value] = 255

    # 寻找前景区域的轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历每个前景轮廓
    for contour in contours:
        # 创建轮廓的掩码
        contour_mask = np.zeros_like(binary_mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # 提取轮廓内部区域
        internal_area = np.logical_and(binary_mask == 0, contour_mask == 255)

        # 检查内部区域是否包含非前景像素
        internal_pixels = mask[internal_area]
        contains_background = any(pixel in background_values for pixel in np.unique(internal_pixels))

        # 如果内部区域包含非前景像素，则将其全部转换为前景
        if contains_background:
            processed_mask[internal_area] = foreground_value

    return processed_mask


def postprocess_mask(mask, min_area=15000, morph_kernel_size=3):
    """
    完整的掩码后处理流程：去除内部区域+形态学操作（直接处理[0,1,2]格式掩码）

    Args:
        mask: 输入掩码图像，值为[0,1,2]
        min_area: 最小连通域面积，小于该值的区域将被移除
        morph_kernel_size: 形态学操作核大小

    Returns:
        处理后的掩码图像，值仍为[0,1,2]
    """
    # 1. 去除前景内部的非前景连通域
    mask = remove_internal_regions(mask)

    # 2. 形态学操作需要二值化
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask == 2] = 255  # 只保留前景类别2

    # 3. 形态学开运算：去除小的噪点
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 4. 连通域分析：去除小面积区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask, connectivity=8)
    processed_binary = np.zeros_like(opened_mask)

    for i in range(1, num_labels):  # 跳过背景(标签0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            processed_binary[labels == i] = 255

    # 5. 转回[0,1,2]格式
    processed_mask = mask.copy()
    processed_mask[processed_binary == 0] = 0  # 移除小的前景区域
    processed_mask[processed_binary == 255] = 2  # 保留大的前景区域

    return processed_mask