import argparse
import logging
import json
from pathlib import Path
from PIL import Image
from typing import Dict


class PngNormalizer:
    """PNG图像归一化器 - 将图像无损缩放至指定尺寸并记录原始尺寸"""

    def __init__(self, input_dir: str, output_dir: str,
                 target_size: int = 512,
                 original_sizes_json: str = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.original_sizes_json = original_sizes_json
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

    def _process_single_image(self, img_path: Path) -> bool:
        """处理单个图像文件"""
        try:
            filename = img_path.name
            self.logger.info(f"处理图像: {filename}")

            # 打开图片并转换为8位灰度图
            with Image.open(img_path) as img:
                if img.mode != 'L':
                    self.logger.info(f"将 {filename} 转换为8位灰度图")
                    img = img.convert('L')  # 'L'模式为8位灰度

                # 记录原始尺寸
                original_width, original_height = img.size
                self.original_sizes[filename] = {
                    "width": original_width,
                    "height": original_height
                }

                # 计算等比例缩放尺寸
                if original_width >= original_height:
                    # 宽是长边
                    scale = self.target_size / original_width
                    new_width = self.target_size
                    new_height = int(original_height * scale)
                else:
                    # 高是长边
                    scale = self.target_size / original_height
                    new_height = self.target_size
                    new_width = int(original_width * scale)

                # 高质量缩放
                resized_img = img.resize((new_width, new_height), resample=Image.LANCZOS)

                # 创建黑色背景画布
                new_img = Image.new('L', (self.target_size, self.target_size), 0)

                # 居中粘贴缩放后的图像
                paste_x = (self.target_size - new_width) // 2
                paste_y = (self.target_size - new_height) // 2
                new_img.paste(resized_img, (paste_x, paste_y))

                # 保存处理后的图像
                output_path = Path(self.output_dir) / filename
                new_img.save(output_path, "PNG", quality=100, compress_level=9)

                self.logger.info(
                    f"已处理: {filename} (原始尺寸: {original_width}x{original_height} -> 新尺寸: {self.target_size}x{self.target_size})")
                return True

        except Exception as e:
            self.logger.error(f"处理 {filename} 时出错: {e}", exc_info=True)
            return False

    def _save_original_sizes(self) -> None:
        """保存原始尺寸信息到JSON文件"""
        if not self.original_sizes_json or not self.original_sizes:
            return

        try:
            with open(self.original_sizes_json, 'w', encoding='utf-8') as f:
                json.dump(self.original_sizes, f, ensure_ascii=False, indent=2)
            self.logger.info(f"原始尺寸信息已保存至: {self.original_sizes_json}")
        except Exception as e:
            self.logger.error(f"保存原始尺寸JSON失败: {e}", exc_info=True)

    def normalize(self) -> Dict[str, int]:
        """执行图像归一化处理"""
        self.logger.info(f"开始图像归一化处理 - 输入目录: {self.input_dir}, 输出目录: {self.output_dir}")
        self.logger.info(f"目标尺寸: {self.target_size}x{self.target_size}")

        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # 获取所有PNG图片
        png_files = list(Path(self.input_dir).glob("*.png"))
        if not png_files:
            self.logger.warning(f"在 {self.input_dir} 中未找到PNG图片")
            return {"processed": 0, "failed": 0}

        self.logger.info(f"找到 {len(png_files)} 张PNG图片")

        # 处理每张图片
        processed_count = 0
        failed_count = 0

        for img_path in png_files:
            success = self._process_single_image(img_path)
            processed_count += 1 if success else 0
            failed_count += 0 if success else 1

        # 保存原始尺寸信息
        self._save_original_sizes()

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
    parser = argparse.ArgumentParser(description='将PNG图片无损缩放至指定尺寸并保存原始尺寸信息')
    parser.add_argument('--input-dir', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output-dir', required=True, help='输出图片文件夹路径')
    parser.add_argument('--target-size', type=int, default=512, help='目标尺寸（宽高相同，默认512）')
    parser.add_argument('--original-sizes', required=True, help='保存原始尺寸信息的JSON文件路径')

    args = parser.parse_args()

    # 创建归一化器实例并执行归一化
    normalizer = PngNormalizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        original_sizes_json=args.original_sizes
    )

    normalizer.normalize()


if __name__ == "__main__":
    main()