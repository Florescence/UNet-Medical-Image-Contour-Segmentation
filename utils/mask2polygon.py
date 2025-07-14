import numpy as np
import cv2
import json
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict


class MaskProcessor:
    """处理医学图像掩码并转换为JSON格式的工具类（覆盖图叠加在原始PNG上）"""

    def __init__(self, input_path: str, output_path: str = None, sizes_json_path: str = None):
        """
        初始化掩码处理器
        :param input_path: 输入掩码文件或目录路径
        :param output_path: 输出路径（默认与输入路径一致）
        :param sizes_json_path: 原始尺寸JSON文件路径（从命令行传入）
        """
        self.input_path = Path(input_path)
        self.output_path = self._get_output_path(output_path)
        self.sizes_json_path = Path(sizes_json_path) if sizes_json_path else None
        self.sizes_data = self._load_sizes_data()  # 一次性加载所有尺寸信息
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _get_output_path(self, output_path: str = None) -> Path:
        """确定输出路径（默认与输入路径一致）"""
        if output_path:
            return Path(output_path)
        if self.input_path.is_file():
            return self.input_path.parent  # 单文件：输出到同目录
        else:
            return self.input_path  # 目录：输出到自身

    def _load_sizes_data(self) -> Dict:
        """从指定JSON文件加载所有图像尺寸信息（一次性加载）"""
        if not self.sizes_json_path or not self.sizes_json_path.exists():
            raise FileNotFoundError(f"尺寸JSON文件不存在: {self.sizes_json_path}")

        with open(self.sizes_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_image_size(self, mask_filename: str) -> Dict[str, int]:
        """从已加载的尺寸数据中获取当前掩码的尺寸"""
        if mask_filename not in self.sizes_data:
            raise KeyError(f"JSON中未找到 {mask_filename} 的尺寸信息")
        return self.sizes_data[mask_filename]

    def _find_original_png(self, base_name: str) -> Path:
        """查找与掩码对应的原始PNG图像（优先同目录，再尝试流程目录）"""
        # 候选路径：优先同目录下的原始PNG
        png_candidates = [
            self.output_path / f"{base_name}.png",  # 同目录
            self.output_path.parent / "1_raw_png" / f"{base_name}.png",  # 流程中的RAW转PNG目录
            self.input_path.parent / f"{base_name}.png"  # 输入掩码的同目录
        ]

        for candidate in png_candidates:
            if candidate.exists() and candidate.suffix.lower() == '.png':
                return candidate
        return None

    def process_mask(self, mask_path: Path) -> bool:
        """处理单个掩码文件，生成JSON和覆盖图（叠加原始PNG）"""
        try:
            mask_filename = mask_path.name
            self.logger.info(f"处理掩码: {mask_filename}")

            # 读取图像原始尺寸
            size_info = self._get_image_size(mask_filename)
            image_width = size_info["width"]
            image_height = size_info["height"]
            self.logger.debug(f"加载原始尺寸: {image_width}x{image_height}")

            # 读取掩码并二值化
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取掩码文件: {mask_path}")

            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # 提取轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt.squeeze(axis=1) for cnt in contours]

            if not contours:
                self.logger.warning(f"未检测到轮廓: {mask_filename}")
                return False

            # 构建JSON数据
            base_name = mask_path.stem
            json_data = {
                "version": "1.0.2.799",
                "imagePath": base_name,
                "imageData": None,
                "flags": {},
                "shapes": [],
                "imageWidth": image_width,
                "imageHeight": image_height
            }

            # 添加所有轮廓点
            for contour in contours:
                points = contour.tolist()
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

            # 保存JSON
            json_path = self.output_path / f"{base_name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"JSON已保存: {json_path}")

            # 生成覆盖图（叠加原始PNG图像）
            self._create_overlay_image(contours, base_name)

            return True

        except Exception as e:
            self.logger.error(f"处理 {mask_path.name} 失败: {str(e)}", exc_info=True)
            return False

    def _create_overlay_image(self, contours: List[np.ndarray], base_name: str) -> None:
        """创建轮廓叠加原始PNG的覆盖图"""
        # 查找原始PNG图像
        original_png = self._find_original_png(base_name)
        if not original_png:
            self.logger.warning(f"未找到原始PNG文件，跳过覆盖图生成: {base_name}.png")
            return

        try:
            # 读取原始PNG并转换为彩色
            original_img = cv2.imread(str(original_png))
            if original_img is None:
                self.logger.warning(f"无法读取原始PNG: {original_png}")
                return

            # 绘制轮廓（转换为3D格式）
            contours_3d = [cnt[:, np.newaxis, :] for cnt in contours]
            cv2.drawContours(original_img, contours_3d, -1, (0, 0, 255), 4)  # 红色轮廓，线宽4

            # 保存覆盖图
            overlay_path = self.output_path / f"{base_name}_contour_overlay.png"
            cv2.imwrite(str(overlay_path), original_img)
            self.logger.info(f"覆盖图已保存: {overlay_path}")

        except Exception as e:
            self.logger.error(f"生成覆盖图失败: {str(e)}")

    def process(self) -> Dict[str, int]:
        """处理输入路径（单文件或目录）"""
        # 收集所有待处理的掩码文件
        mask_files = []
        if self.input_path.is_file():
            if self.input_path.suffix.lower() == '.png':
                mask_files = [self.input_path]
            else:
                self.logger.error("输入文件不是PNG格式")
                return {"total": 0, "success": 0, "failed": 0}
        else:
            mask_files = list(self.input_path.glob("*.png"))
            if not mask_files:
                self.logger.warning(f"目录中未找到PNG文件: {self.input_path}")
                return {"total": 0, "success": 0, "failed": 0}

        self.logger.info(f"共发现 {len(mask_files)} 个掩码文件")

        # 处理所有文件
        success_count = 0
        for mask_file in mask_files:
            if self.process_mask(mask_file):
                success_count += 1

        result = {
            "total": len(mask_files),
            "success": success_count,
            "failed": len(mask_files) - success_count
        }
        self.logger.info(f"处理完成 - 总计: {result['total']}, 成功: {result['success']}, 失败: {result['failed']}")
        return result


def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 命令行参数
    parser = argparse.ArgumentParser(description="将掩码转换为轮廓JSON（覆盖图叠加原始PNG）")
    parser.add_argument('-i', '--input', required=True, help="输入掩码文件（.png）或目录")
    parser.add_argument('-o', '--output', help="输出路径（默认与输入一致）")
    parser.add_argument('-j', '--json', required=True, help="记录原始尺寸的JSON文件路径")

    args = parser.parse_args()

    # 初始化处理器并执行
    try:
        processor = MaskProcessor(
            input_path=args.input,
            output_path=args.output,
            sizes_json_path=args.json
        )
        processor.process()
    except Exception as e:
        logging.error(f"处理失败: {str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()