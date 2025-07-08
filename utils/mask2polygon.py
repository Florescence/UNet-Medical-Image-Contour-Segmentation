import numpy as np
import cv2
import json
import os
import argparse


class MaskProcessor:
    """处理医学图像掩码并转换为JSON格式的工具类"""

    def __init__(self, mask_dir, output_dir=None, image_width=4267, image_height=4267):
        """初始化掩码处理器"""
        self.mask_dir = mask_dir
        self.output_dir = output_dir or mask_dir
        self.image_width = image_width
        self.image_height = image_height

        # 创建输出目录（如果不存在）
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def read_raw_image(file_path, width=4267, height=4267, dtype=np.uint16):
        """读取RAW格式的医学图像"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            img_data = np.frombuffer(raw_data, dtype=dtype)
            img_data = img_data[:width * height].reshape((height, width))
            return img_data
        except Exception as e:
            print(f"读取RAW图像失败: {e}")
            return None

    @staticmethod
    def convert_to_uint8(img):
        """将图像转换为uint8格式以便显示或保存"""
        if img.dtype == np.uint8:
            return img
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    def process_mask(self, mask_file, max_points, simplify=True, tolerance=2.0, min_points=10, overlay_raw=True):
        """处理单个掩码图像并转换为JSON格式"""
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码图像: {mask_file}")

        base_name = os.path.splitext(os.path.basename(mask_file))[0].replace("_pred_mask", "")
        image_width = self.image_width
        image_height = self.image_height

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 提取轮廓并转换为二维坐标
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt.squeeze(axis=1) for cnt in contours]  # 转为二维数组 (n, 2)

        json_data = {
            "version": "1.0.2.799",
            "imagePath": base_name,
            "imageData": None,
            "flags": {},
            "shapes": [],
            "imageWidth": image_width,
            "imageHeight": image_height
        }

        if not contours:
            print(f"警告: 未找到轮廓: {mask_file}")
            return json_data

        for i, contour in enumerate(contours):
            simplified_contour = self._simplify_contour(contour, max_points, simplify, tolerance, min_points)

            # 使用二维点坐标
            points = simplified_contour.tolist()
            json_data["shapes"].append({
                "label": 1,
                "labelIndex": 0,
                "points": points,
                "shape_type": "polygon",
                "description": "",
                "mask": None,
                "group_id": None,
                "flags": {}
            })

        # 叠加轮廓并保存图像
        if overlay_raw:
            self._create_overlay_image(contours, base_name, image_width, image_height)

        # 保存JSON文件
        output_path = os.path.join(self.output_dir, f"{base_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"JSON已保存至: {output_path}")
        return json_data

    def _simplify_contour(self, contour, max_points, simplify, tolerance, min_points):
        """简化轮廓点集，平衡精度和点数量"""
        if simplify:
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter > 0:
                epsilon = max(0.001, min(0.05, tolerance / 100.0)) * perimeter
                simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)

                if len(simplified_contour) < min_points and len(contour) >= min_points:
                    step = max(1, len(contour) // min_points)
                    sampled_points = contour[::step]

                    if len(sampled_points) > 0:
                        first_point = sampled_points[0]
                        last_point = sampled_points[-1]

                        if not np.allclose(first_point, last_point, atol=1.0):
                            sampled_points = np.vstack([sampled_points, first_point])

                    simplified_contour = sampled_points
            else:
                simplified_contour = contour
        else:
            distance_weight = 0.2
            if len(contour) <= max_points:
                simplified_contour = contour
            else:
                # 混合采样策略：距离优先 + 均匀分布
                distances = np.zeros(len(contour))
                for j in range(len(contour)):
                    prev_idx = (j - 1) % len(contour)
                    curr_idx = j
                    next_idx = (j + 1) % len(contour)

                    p1 = contour[prev_idx]
                    p2 = contour[curr_idx]
                    p3 = contour[next_idx]

                    dist_prev = np.linalg.norm(p2 - p1)
                    dist_next = np.linalg.norm(p3 - p2)

                    distances[j] = dist_prev + dist_next

                # 按距离排序
                distance_indices = np.argsort(distances)[::-1]
                distance_count = int(max_points * distance_weight)
                distance_points = distance_indices[:distance_count]

                # 均匀分布采样
                uniform_count = max_points - distance_count
                uniform_step = max(1, len(contour) // uniform_count)
                uniform_indices = np.arange(0, len(contour), uniform_step)

                # 合并采样结果并去重
                all_indices = np.unique(np.concatenate([distance_points, uniform_indices]))

                # 确保包含首尾点
                if 0 not in all_indices:
                    all_indices = np.insert(all_indices, 0, 0)

                if len(contour) - 1 not in all_indices:
                    all_indices = np.append(all_indices, len(contour) - 1)

                # 按索引提取点
                simplified_contour = contour[np.sort(all_indices)]

                # 确保首尾点闭合
                if len(simplified_contour) > 0 and not np.allclose(simplified_contour[0], simplified_contour[-1],
                                                                   atol=1.0):
                    simplified_contour = np.vstack([simplified_contour, simplified_contour[0]])

        return simplified_contour

    def _create_overlay_image(self, contours, base_name, image_width, image_height):
        """创建并保存带有轮廓叠加的图像"""
        # 从掩码目录的上级目录查找RAW文件（适配seg_main.py的目录结构）
        raw_dir = os.path.dirname(os.path.dirname(self.mask_dir))  # 调整路径层级适配整体流程
        raw_path = os.path.join(raw_dir, "1_raw_png", f"{base_name}.raw")  # 假设RAW文件在1_raw_png目录

        # 兼容原始路径查找逻辑
        if not os.path.exists(raw_path):
            raw_path = os.path.join(os.path.dirname(self.mask_dir), f"{base_name}.raw")

        if os.path.exists(raw_path):
            try:
                raw_img = self.read_raw_image(raw_path, width=image_width, height=image_height)
                if raw_img is None:
                    return

                raw_8bit = self.convert_to_uint8(raw_img)
                raw_color = cv2.cvtColor(raw_8bit, cv2.COLOR_GRAY2BGR)

                # 转换轮廓为三维格式供OpenCV绘制
                contours_3d = [cnt[:, np.newaxis, :] for cnt in contours]
                cv2.drawContours(raw_color, contours_3d, -1, (0, 0, 255), 4)

                overlay_path = os.path.join(self.output_dir, f"{base_name}_contour_overlay.png")
                cv2.imwrite(overlay_path, raw_color)
                print(f"轮廓叠加图像已保存至: {overlay_path}")
            except Exception as e:
                print(f"生成叠加图像失败: {e}")

    def process_dataset(self, max_points, simplify=True, tolerance=2.0, min_points=10):
        """处理整个掩码数据集"""
        # 收集所有掩码文件
        mask_files = []
        for root, _, files in os.walk(self.mask_dir):
            for file in files:
                if file.lower().endswith('mask.png'):
                    mask_files.append(os.path.join(root, file))

        processed_count = 0

        for mask_file in mask_files:
            try:
                print(f"\n处理文件: {os.path.basename(mask_file)}")
                self.process_mask(
                    mask_file,
                    max_points,
                    simplify=simplify,
                    tolerance=tolerance,
                    min_points=min_points
                )
                processed_count += 1
            except Exception as e:
                print(f"处理文件 {mask_file} 时出错: {e}")

        print(f"处理完成! 共处理了 {processed_count} 个掩码图像")


def main():
    # 仅允许传递宽高参数，其他参数在main中写死
    parser = argparse.ArgumentParser(description="将掩码转换为轮廓坐标JSON")
    parser.add_argument("--mask-dir", required=True, help="掩码图像目录")
    parser.add_argument("--output-dir", help="输出JSON和覆盖图的目录")
    parser.add_argument("--width", "-w", type=int, required=True, help="图像宽度")
    parser.add_argument("--height", "-h", type=int, required=True, help="图像高度")

    args = parser.parse_args()

    # 写死的参数
    MAX_POINTS = 500
    SIMPLIFY = False
    TOLERANCE = 1.0
    MIN_POINTS = 10

    # 创建处理器并处理
    processor = MaskProcessor(
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        image_width=args.width,
        image_height=args.height
    )
    processor.process_dataset(
        max_points=MAX_POINTS,
        simplify=SIMPLIFY,
        tolerance=TOLERANCE,
        min_points=MIN_POINTS
    )


if __name__ == "__main__":
    main()