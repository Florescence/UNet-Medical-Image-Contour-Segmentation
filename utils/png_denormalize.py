import argparse
import logging
import json
from pathlib import Path
from PIL import Image
from typing import Dict, Union


class PngDenormalizer:
    """PNG图像反归一化器 - 支持单张/目录输入，恢复图像到原始尺寸"""

    def __init__(self, input_path: str, output_path: str = None,
                 original_sizes_json: str = None, target_size: int = 512):
        """
        初始化反归一化器
        :param input_path: 输入PNG文件或目录路径
        :param output_path: 输出路径（可选，默认与输入相同）
        :param original_sizes_json: 原始尺寸JSON路径（可选，自动推断）
        :param target_size: 归一化目标尺寸（默认512）
        """
        self.input_path = Path(input_path)
        self.output_path = self._get_output_path(output_path)
        self.original_sizes_json = self._get_json_path(original_sizes_json)
        self.target_size = target_size
        self.original_sizes = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def _get_output_path(self, output_path: Union[str, None]) -> Path:
        """获取输出路径（单文件/目录自动处理）"""
        if output_path:
            return Path(output_path)

        # 默认与输入路径相同
        if self.input_path.is_file():
            return self.input_path.parent  # 单文件：输出到同目录
        else:
            return self.input_path  # 目录：输出到自身

    def _get_json_path(self, json_path: Union[str, None]) -> Path:
        """获取原始尺寸JSON路径（自动推断）"""
        if json_path:
            return Path(json_path)

        # 自动推断JSON路径
        if self.input_path.is_file():
            # 单文件：JSON与文件同目录，命名为"文件名_sizes.json"
            return self.input_path.parent / f"{self.input_path.stem}_sizes.json"
        else:
            # 目录：JSON在目录内，命名为"original_sizes.json"
            return self.input_path / "original_sizes.json"

    def _load_original_sizes(self) -> bool:
        """加载原始尺寸信息"""
        try:
            with open(self.original_sizes_json, 'r', encoding='utf-8') as f:
                self.original_sizes = json.load(f)
            self.logger.info(f"成功加载 {len(self.original_sizes)} 个原始尺寸记录")
            return True
        except Exception as e:
            self.logger.error(f"加载原始尺寸JSON失败: {e}")
            return False

    def _process_single_image(self, img_path: Path) -> bool:
        """处理单个图像文件"""
        try:
            filename = img_path.name
            self.logger.info(f"处理图像: {filename}")

            # 检查是否有原始尺寸记录
            if filename not in self.original_sizes:
                self.logger.warning(f"找不到 {filename} 的原始尺寸信息，跳过处理")
                return False

            # 获取原始尺寸
            orig_width = self.original_sizes[filename]["width"]
            orig_height = self.original_sizes[filename]["height"]

            # 打开归一化后的图片
            with Image.open(img_path) as img:
                # 计算归一化时的缩放比例和填充
                if orig_width >= orig_height:
                    scale = self.target_size / orig_width
                    new_width = self.target_size
                    new_height = int(orig_height * scale)
                    padding_x = 0
                    padding_y = (self.target_size - new_height) // 2
                else:
                    scale = self.target_size / orig_height
                    new_height = self.target_size
                    new_width = int(orig_width * scale)
                    padding_x = (self.target_size - new_width) // 2
                    padding_y = 0

                # 裁剪黑边
                cropped_img = img.crop((padding_x, padding_y, padding_x + new_width, padding_y + new_height))

                # 缩放回原始尺寸
                final_img = cropped_img.resize(
                    (orig_width, orig_height),
                    resample=Image.LANCZOS  # 使用高质量插值
                )

                # 保存反归一化后的图片
                output_path = self.output_path / filename
                final_img.save(output_path, "PNG", quality=100, compress_level=9)

                self.logger.info(f"已处理: {filename} (恢复至原始尺寸: {orig_width}x{orig_height})")
                return True

        except Exception as e:
            self.logger.error(f"处理 {filename} 时出错: {e}")
            return False

    def denormalize(self) -> Dict[str, int]:
        """执行图像反归一化处理"""
        self.logger.info(f"开始图像反归一化处理")
        self.logger.info(f"输入: {self.input_path}")
        self.logger.info(f"输出: {self.output_path}")
        self.logger.info(f"原始尺寸信息: {self.original_sizes_json}")
        self.logger.info(f"目标尺寸: {self.target_size}x{self.target_size}")

        # 加载原始尺寸信息
        if not self._load_original_sizes():
            return {"processed": 0, "failed": 0, "total": 0}

        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有待处理的PNG文件
        if self.input_path.is_file():
            # 单文件处理
            if self.input_path.suffix.lower() != '.png':
                self.logger.error(f"输入文件不是PNG格式: {self.input_path}")
                return {"processed": 0, "failed": 0, "total": 0}
            png_files = [self.input_path]
        else:
            # 目录处理
            png_files = list(self.input_path.glob("*.png"))
            if not png_files:
                self.logger.warning(f"在 {self.input_path} 中未找到PNG图片")
                return {"processed": 0, "failed": 0, "total": 0}

        self.logger.info(f"找到 {len(png_files)} 张PNG图片")

        # 处理每张图片
        processed_count = 0
        failed_count = 0

        for img_path in png_files:
            success = self._process_single_image(img_path)
            processed_count += 1 if success else 0
            failed_count += 0 if success else 1

        result = {
            "processed": processed_count,
            "failed": failed_count,
            "total": processed_count + failed_count
        }

        self.logger.info(
            f"处理完成 - 成功: {processed_count}, 失败: {failed_count}, 总计: {processed_count + failed_count}")
        return result


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将归一化的PNG图片反归一化回原始尺寸')
    parser.add_argument('-i', '--input', required=True, help='输入PNG文件或目录路径')
    parser.add_argument('-o', '--output', help='输出路径（可选，默认与输入相同）')
    parser.add_argument('-j', '--json', help='原始尺寸JSON文件路径（可选，自动推断）')
    parser.add_argument('-s', '--size', type=int, default=512, help='归一化目标尺寸（默认512）')

    args = parser.parse_args()

    # 创建反归一化器实例并执行反归一化
    denormalizer = PngDenormalizer(
        input_path=args.input,
        output_path=args.output,
        original_sizes_json=args.json,
        target_size=args.size
    )

    denormalizer.denormalize()


if __name__ == "__main__":
    main()