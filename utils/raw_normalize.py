import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, Union


class RawNormalizer:
    """RAW图像归一化器 - 直接处理RAW数据并生成归一化的张量"""

    def __init__(self, input_path: str, output_path: str = None,
                 width: int = None, height: int = None,
                 bit_depth: int = 16, channel_order: str = 'RGGB'):
        """
        初始化RAW图像归一化器
        :param input_path: 输入RAW文件路径（单张）或目录路径
        :param output_path: 输出路径（可选，默认与输入路径相同）
        :param width: RAW图像宽度（像素）
        :param height: RAW图像高度（像素）
        :param bit_depth: RAW位深度（默认16位）
        :param channel_order: 拜耳模式（默认RGGB）
        """
        self.input_path = Path(input_path)
        self.output_path = self._get_default_output_path(output_path)
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.channel_order = channel_order
        self.target_size = 512  # 固定目标尺寸
        self.original_sizes = {}  # 存储原始尺寸信息
        self.logger = self._setup_logger()

        # 验证参数
        if self.input_path.is_file() and (width is None or height is None):
            raise ValueError("处理单张RAW文件时必须指定width和height参数")

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
        """获取默认输出路径"""
        if output_path:
            return Path(output_path)

        # 若未指定输出路径，使用输入路径
        if self.input_path.is_file():
            return self.input_path.parent  # 单张文件：输出到同目录
        else:
            return self.input_path  # 目录：输出到自身目录

    def _get_json_path(self) -> Path:
        """获取原始尺寸JSON文件的默认路径"""
        if self.input_path.is_file():
            # 单张文件：JSON与文件同目录，以文件名为前缀
            return self.output_path / f"{self.input_path.stem}_sizes.json"
        else:
            # 目录：JSON在输出目录下，命名为original_sizes.json
            return self.output_path / "original_sizes.json"

    def _read_raw_file(self, raw_path: Path) -> np.ndarray:
        """读取RAW文件并转换为numpy数组"""
        try:
            # 读取二进制数据
            with open(raw_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint16)

            # 重塑为图像形状
            img = data.reshape((self.height, self.width))
            return img

        except Exception as e:
            self.logger.error(f"读取RAW文件失败: {e}", exc_info=True)
            raise

    def _process_single_raw(self, raw_path: Path) -> bool:
        """处理单个RAW文件"""
        try:
            filename = raw_path.name
            self.logger.info(f"处理RAW文件: {filename}")

            # 读取RAW数据
            raw_data = self._read_raw_file(raw_path)
            original_height, original_width = raw_data.shape

            # 记录原始尺寸
            self.original_sizes[filename] = {
                "width": original_width,
                "height": original_height
            }

            # 归一化处理
            # 1. 转换为浮点数并归一化到 [0,1]
            normalized = raw_data.astype(np.float32) / (2 ** self.bit_depth - 1)

            # 2. 等比例缩放至目标尺寸（长边=512）
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

            # 使用简单的最近邻插值进行缩放（可替换为更高级的插值方法）
            scaled = np.zeros((new_height, new_width), dtype=np.float32)
            for y in range(new_height):
                for x in range(new_width):
                    src_y = min(int(y / scale), original_height - 1)
                    src_x = min(int(x / scale), original_width - 1)
                    scaled[y, x] = normalized[src_y, src_x]

            # 3. 保存为.npy格式（便于模型直接加载）
            output_npy_path = self.output_path / f"{raw_path.stem}.npy"
            np.save(output_npy_path, scaled)

            self.logger.info(
                f"已处理: {filename} (原始尺寸: {original_width}x{original_height} -> 缩放后: {new_width}x{new_height})")
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
        """执行RAW图像归一化处理"""
        self.logger.info(f"开始RAW图像归一化处理 - 目标尺寸: 长边={self.target_size}")
        self.logger.info(f"输入: {self.input_path}, 输出: {self.output_path}")
        self.logger.info(
            f"图像参数: 宽度={self.width}, 高度={self.height}, 位深度={self.bit_depth}, 拜耳模式={self.channel_order}")

        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 收集需要处理的RAW文件
        if self.input_path.is_file():
            # 处理单张文件
            raw_files = [self.input_path] if self.input_path.suffix.lower() in ['.raw', '.data'] else []
        else:
            # 处理目录下所有RAW
            raw_files = list(self.input_path.glob("*.raw")) + list(self.input_path.glob("*.data"))

        if not raw_files:
            self.logger.warning(f"未找到RAW文件 (路径: {self.input_path})")
            return {"processed": 0, "failed": 0, "total": 0}

        self.logger.info(f"找到 {len(raw_files)} 张RAW文件")

        # 处理每张RAW
        processed_count = 0
        failed_count = 0

        for raw_path in raw_files:
            # 对于目录处理模式，动态获取每张图像的尺寸（如果提供了尺寸文件）
            if self.input_path.is_dir():
                # 此处可添加逻辑从单独的配置文件读取每张图像的尺寸
                # 目前使用命令行传入的统一尺寸
                pass

            success = self._process_single_raw(raw_path)
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
    parser = argparse.ArgumentParser(description='将RAW图像归一化并生成可用于训练的张量')
    parser.add_argument('--input', help='输入RAW文件路径或包含RAW文件的目录', required=True)
    parser.add_argument('--output', '-o', help='输出路径（可选，默认与输入路径相同）')
    parser.add_argument('--width', type=int, help='RAW图像宽度（像素）', required=True)
    parser.add_argument('--height', type=int, help='RAW图像高度（像素）', required=True)
    parser.add_argument('--bit-depth', type=int, default=16, help='RAW位深度（默认16位）')
    parser.add_argument('--channel-order', default='RGGB', help='拜耳模式（默认RGGB）')

    args = parser.parse_args()

    # 创建归一化器实例并执行归一化
    normalizer = RawNormalizer(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
        bit_depth=args.bit_depth,
        channel_order=args.channel_order
    )

    normalizer.normalize()


if __name__ == "__main__":
    main()