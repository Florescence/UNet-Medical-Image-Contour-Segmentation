import argparse
import logging
import json
from pathlib import Path
from PIL import Image
from typing import Dict


class PngDenormalizer:
    """PNG图像反归一化器 - 将归一化图像恢复到原始尺寸"""

    def __init__(self, input_dir: str, output_dir: str,
                 original_sizes_json: str,
                 target_size: int = 512):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.original_sizes_json = original_sizes_json
        self.target_size = target_size
        self.original_sizes = {}  # 存储原始尺寸信息
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

    def _load_original_sizes(self) -> bool:
        """加载原始尺寸信息"""
        try:
            with open(self.original_sizes_json, 'r', encoding='utf-8') as f:
                self.original_sizes = json.load(f)
            self.logger.info(f"成功加载 {len(self.original_sizes)} 个原始尺寸记录")
            return True
        except Exception as e:
            self.logger.error(f"加载原始尺寸JSON失败: {e}", exc_info=True)
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
                    # 宽是长边
                    scale = self.target_size / orig_width
                    new_width = self.target_size
                    new_height = int(orig_height * scale)
                    padding_x = 0
                    padding_y = (self.target_size - new_height) // 2
                else:
                    # 高是长边
                    scale = self.target_size / orig_height
                    new_height = self.target_size
                    new_width = int(orig_width * scale)
                    padding_x = (self.target_size - new_width) // 2
                    padding_y = 0

                # 计算裁剪区域（去除黑边）
                crop_box = (
                    padding_x,  # 左边界
                    padding_y,  # 上边界
                    padding_x + new_width,  # 右边界
                    padding_y + new_height  # 下边界
                )

                # 裁剪黑边
                cropped_img = img.crop(crop_box)

                # 缩放回原始尺寸
                final_img = cropped_img.resize(
                    (orig_width, orig_height),
                    resample=Image.LANCZOS  # 使用高质量插值算法
                )

                # 保存反归一化后的图片
                output_path = Path(self.output_dir) / filename
                final_img.save(output_path, "PNG", quality=100, compress_level=9)

                self.logger.info(f"已处理: {filename} (恢复至原始尺寸: {orig_width}x{orig_height})")
                return True

        except Exception as e:
            self.logger.error(f"处理 {filename} 时出错: {e}", exc_info=True)
            return False

    def denormalize(self) -> Dict[str, int]:
        """执行图像反归一化处理"""
        self.logger.info(f"开始图像反归一化处理 - 输入目录: {self.input_dir}, 输出目录: {self.output_dir}")
        self.logger.info(f"目标尺寸: {self.target_size}x{self.target_size}, 原始尺寸信息: {self.original_sizes_json}")

        # 加载原始尺寸信息
        if not self._load_original_sizes():
            return {"processed": 0, "failed": 0, "total": 0}

        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 获取所有PNG图片
        png_files = list(Path(self.input_dir).glob("*.png"))
        if not png_files:
            self.logger.warning(f"在 {self.input_dir} 中未找到PNG图片")
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
    parser.add_argument('--input-dir', required=True, help='输入归一化图片文件夹路径')
    parser.add_argument('--output-dir', required=True, help='输出反归一化图片文件夹路径')
    parser.add_argument('--original-sizes', required=True, help='原始尺寸信息的JSON文件路径')
    parser.add_argument('--target-size', type=int, default=512, help='归一化时使用的目标尺寸（默认512）')

    args = parser.parse_args()

    # 创建反归一化器实例并执行反归一化
    denormalizer = PngDenormalizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        original_sizes_json=args.original_sizes,
        target_size=args.target_size
    )

    denormalizer.denormalize()


if __name__ == "__main__":
    main()