import argparse
import logging
import json
from pathlib import Path
from PIL import Image
from typing import Dict, Union


class PngNormalizer:
    """PNG图像归一化器 - 固定将图像缩放至512x512，支持单张/目录处理"""

    def __init__(self, input_path: str, output_path: str = None):
        """
        初始化归一化器
        :param input_path: 输入图片路径（单张PNG）或目录路径
        :param output_path: 输出路径（可选，默认与输入路径相同）
        """
        self.input_path = Path(input_path)
        self.output_path = self._get_default_output_path(output_path)
        self.target_size = 512  # 固定目标尺寸为512x512
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

    def _get_default_output_path(self, output_path: Union[str, None]) -> Path:
        """获取默认输出路径（与输入路径相同）"""
        if output_path:
            return Path(output_path)

        # 若未指定输出路径，使用输入路径
        if self.input_path.is_file():
            return self.input_path.parent  # 单张图片：输出到同目录
        else:
            return self.input_path  # 目录：输出到自身目录

    def _get_json_path(self) -> Path:
        """获取原始尺寸JSON文件的默认路径"""
        if self.input_path.is_file():
            # 单张图片：JSON与图片同目录，以图片名为前缀
            return self.output_path / f"{self.input_path.stem}_sizes.json"
        else:
            # 目录：JSON在输出目录下，命名为original_sizes.json
            return self.output_path / "original_sizes.json"

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
                output_img_path = self.output_path / filename
                new_img.save(output_img_path, "PNG", quality=100, compress_level=9)

                self.logger.info(
                    f"已处理: {filename} (原始尺寸: {original_width}x{original_height} -> 新尺寸: {self.target_size}x{self.target_size})")
                return True

        except Exception as e:
            self.logger.error(f"处理 {filename} 时出错: {e}", exc_info=True)
            return False

    def _save_original_sizes(self) -> None:
        """保存原始尺寸信息到JSON文件"""
        if not self.original_sizes:
            self.logger.warning("没有需要保存的原始尺寸信息")
            return

        json_path = self._get_json_path()
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.original_sizes, f, ensure_ascii=False, indent=2)
            self.logger.info(f"原始尺寸信息已保存至: {json_path}")
        except Exception as e:
            self.logger.error(f"保存原始尺寸JSON失败: {e}", exc_info=True)

    def normalize(self) -> Dict[str, int]:
        """执行图像归一化处理"""
        self.logger.info(f"开始图像归一化处理 - 目标尺寸: {self.target_size}x{self.target_size}")
        self.logger.info(f"输入: {self.input_path}, 输出: {self.output_path}")

        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 收集需要处理的图片
        if self.input_path.is_file():
            # 处理单张图片
            png_files = [self.input_path] if self.input_path.suffix.lower() == '.png' else []
        else:
            # 处理目录下所有PNG
            png_files = list(self.input_path.glob("*.png"))

        if not png_files:
            self.logger.warning(f"未找到PNG图片 (路径: {self.input_path})")
            return {"processed": 0, "failed": 0, "total": 0}

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
    parser = argparse.ArgumentParser(description='将PNG图片归一化至512x512并记录原始尺寸')
    parser.add_argument('--input', help='输入PNG图片路径或包含PNG图片的目录', required=True)
    parser.add_argument('--output', '-o', help='输出路径（可选，默认与输入路径相同）')

    args = parser.parse_args()

    # 创建归一化器实例并执行归一化
    normalizer = PngNormalizer(
        input_path=args.input,
        output_path=args.output
    )

    normalizer.normalize()


if __name__ == "__main__":
    main()