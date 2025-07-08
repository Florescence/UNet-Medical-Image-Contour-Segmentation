import os
import numpy as np
import imageio
import logging
import argparse
from typing import Tuple


class RawToPngConverter:
    """将16-bit RAW图像转换为PNG格式的转换器"""

    def __init__(self, input_path: str, output_dir: str = None,
                 width: int = None, height: int = None,
                 window_center: int = None, window_width: int = None):
        self.input_path = input_path
        self.output_dir = output_dir or os.path.dirname(input_path)
        self.width = width
        self.height = height
        self.window_center = window_center
        self.window_width = window_width
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def _read_16bit_raw(self, raw_path: str) -> np.ndarray:
        """读取16-bit RAW图像数据"""
        # 检查文件大小
        file_size = os.path.getsize(raw_path)
        expected_size = self.width * self.height * 2  # 16-bit = 2字节/像素

        if file_size != expected_size:
            self.logger.warning(
                f"文件大小不匹配：实际 {file_size} 字节，预期 {expected_size} 字节（{self.width}x{self.height}）")

        # 读取二进制数据
        with open(raw_path, 'rb') as f:
            data = f.read()

        # 转换为numpy数组
        try:
            img_array = np.frombuffer(data, dtype=np.uint16).reshape((self.height, self.width))
            return img_array
        except ValueError as e:
            self.logger.error(f"数据解析失败：{str(e)}")
            raise

    def _apply_windowing(self, img_array: np.ndarray) -> np.ndarray:
        """应用窗宽窗位计算"""
        # 计算窗宽窗位的上下限
        window_min = self.window_center - self.window_width // 2
        window_max = self.window_center + self.window_width // 2

        # 裁剪到窗宽范围内
        img_clipped = np.clip(img_array, window_min, window_max)

        # 映射到0-255
        img_scaled = ((img_clipped - window_min) / (window_max - window_min) * 255).astype(np.uint8)
        return img_scaled

    def convert_single_file(self, raw_path: str) -> bool:
        """转换单个RAW文件为PNG"""
        try:
            filename = os.path.basename(raw_path)
            self.logger.info(f"开始处理 {filename}")

            # 读取16-bit RAW数据
            img_16bit = self._read_16bit_raw(raw_path)

            # 应用窗宽窗位并转换为8-bit
            img_8bit = self._apply_windowing(img_16bit)

            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 生成输出路径
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(self.output_dir, output_filename)

            # 保存为PNG
            imageio.imwrite(output_path, img_8bit, format='PNG')

            self.logger.info(f"{filename} 处理完成，保存至 {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"{filename} 转换失败：{str(e)}", exc_info=True)
            return False

    def convert(self) -> Tuple[int, int]:
        """执行转换（处理单个文件或目录）"""
        converted_count = 0
        failed_count = 0

        self.logger.info(f"开始转换：输入路径={self.input_path}，输出目录={self.output_dir}")
        self.logger.info(f"图像尺寸：{self.width}x{self.height}，窗宽窗位：{self.window_center}/{self.window_width}")

        # 处理单个文件
        if os.path.isfile(self.input_path) and self.input_path.lower().endswith('.raw'):
            success = self.convert_single_file(self.input_path)
            converted_count += 1 if success else 0
            failed_count += 0 if success else 1

        # 处理目录
        elif os.path.isdir(self.input_path):
            # 遍历目录下所有RAW文件
            raw_files = [
                f for f in os.listdir(self.input_path)
                if os.path.isfile(os.path.join(self.input_path, f)) and f.lower().endswith('.raw')
            ]

            if not raw_files:
                self.logger.warning(f"目录 {self.input_path} 中未找到RAW文件")
                return 0, 0

            self.logger.info(f"找到 {len(raw_files)} 个RAW文件，开始批量处理")

            for filename in raw_files:
                raw_path = os.path.join(self.input_path, filename)
                success = self.convert_single_file(raw_path)
                converted_count += 1 if success else 0
                failed_count += 0 if success else 1
        else:
            self.logger.error(f"输入路径无效：{self.input_path}（不是文件或目录）")
            return 0, 0

        self.logger.info(f"处理完成：成功={converted_count}，失败={failed_count}")
        return converted_count, failed_count


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将16-bit RAW图像转换为PNG格式')
    parser.add_argument('input', help='输入RAW文件路径或包含RAW文件的目录')
    parser.add_argument('--output', '-o', help='输出目录路径（默认与输入同目录）', default=None)
    parser.add_argument('--width', '-w', type=int, required=True, help='图像宽度（像素）')
    parser.add_argument('--height', '-h', type=int, required=True, help='图像高度（像素）')
    parser.add_argument('--center', '-c', type=int, required=True, help='窗位（window center）')
    parser.add_argument('--width-window', '-ww', type=int, required=True, help='窗宽（window width）')

    args = parser.parse_args()

    # 创建转换器实例并执行转换
    converter = RawToPngConverter(
        input_path=args.input,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        window_center=args.center,
        window_width=args.width_window
    )

    converter.convert()


if __name__ == "__main__":
    main()