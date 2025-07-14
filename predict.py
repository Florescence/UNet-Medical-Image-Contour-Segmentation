import os
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from unet import UNet  # 仅用于.pth模型，.pt模型不需要
from utils.utils import plot_img_and_mask
from utils.post_process import postprocess_mask


def predict_img(model, full_img, device):
    """执行模型预测（兼容.pth和.pt模型）"""
    model.eval()  # 确保模型处于推理模式
    # 预处理输入（与原逻辑一致）
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale=1, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

    with torch.no_grad(), torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
        # 对TorchScript模型和普通模型，forward调用方式一致
        mask_pred = model(img)
        # 确保输出尺寸与原图一致（与原逻辑一致）
        mask_pred = F.interpolate(mask_pred, (full_img.size[1], full_img.size[0]), mode='bilinear')
        mask_pred_indices = mask_pred.argmax(dim=1).squeeze(0)

    return mask_pred_indices.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='预测多分类掩码（兼容.pth和.pt模型）')
    parser.add_argument('--model', '-m', required=True, help='模型文件路径（支持.pth或.pt）')
    parser.add_argument('--input', '-i', required=True, help='输入图像文件或目录路径')
    parser.add_argument('--output', '-o', help='输出目录路径（可选，默认覆盖原图）')
    parser.add_argument('--viz', '-v', action='store_true', default=False, help='可视化预测结果')
    parser.add_argument('--no-save', '-n', action='store_true', default=False, help='不保存输出掩码')
    parser.add_argument('--postprocess', '-p', action='store_true', default=True, help='应用后处理')
    return parser.parse_args()


def get_output_path(args, input_file):
    """生成输出路径（支持目录输出和自动覆盖）"""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if args.output is None:
        return os.path.join(os.path.dirname(input_file), f"{base_name}.png")
    os.makedirs(args.output, exist_ok=True)
    return os.path.join(args.output, f"{base_name}.png")


def mask_to_image(mask: np.ndarray):
    """将多分类掩码转为可视化图像"""
    mask_vis = np.zeros_like(mask, dtype=np.uint8)
    mask_vis[mask == 0] = 0
    mask_vis[mask == 1] = 128
    mask_vis[mask == 2] = 255
    return Image.fromarray(mask_vis)


def process_directory(input_dir):
    """递归处理目录中的所有图像文件"""
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 处理输入路径（文件或目录）
    if os.path.isdir(args.input):
        in_files = process_directory(args.input)
        logging.info(f"在目录中找到 {len(in_files)} 个图像文件")
        if not in_files:
            logging.error(f"目录 {args.input} 中未找到图像文件")
            exit(1)
    else:
        if not os.path.isfile(args.input):
            logging.error(f"输入文件不存在: {args.input}")
            exit(1)
        in_files = [args.input]

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')

    # 加载模型（核心修改：兼容.pth和.pt）
    model_path = args.model
    try:
        if model_path.endswith('.pt'):
            # 加载TorchScript模型（.pt）
            logging.info(f'加载TorchScript模型: {model_path}')
            model = torch.jit.load(model_path, map_location=device)
            model.to(device)
            model.eval()  # 显式设置为推理模式
        elif model_path.endswith('.pth'):
            # 加载传统权重文件（.pth）
            logging.info(f'加载传统模型权重: {model_path}')
            model = UNet(n_channels=1, n_classes=3, bilinear=False)  # 需与原模型结构一致
            model.to(device)
            state_dict = torch.load(model_path, map_location=device)
            if 'mask_values' in state_dict:
                del state_dict['mask_values']
            model.load_state_dict(state_dict)
            model.eval()
        else:
            logging.error(f"不支持的模型格式: {model_path}（仅支持.pth和.pt）")
            exit(1)
        logging.info('模型加载完成!')
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}", exc_info=True)
        exit(1)

    # 处理每张图像
    for in_file in in_files:
        logging.info(f'处理图像: {in_file}')
        try:
            img = Image.open(in_file).convert('L')  # 转为灰度图（与C++处理一致）

            # 模型预测（兼容两种模型）
            mask_pred = predict_img(model, img, device)

            # 应用后处理
            if args.postprocess:
                mask_pred = postprocess_mask(mask_pred)
                logging.info('已应用后处理')

            # 保存结果
            if not args.no_save:
                out_file = get_output_path(args, in_file)
                result = mask_to_image(mask_pred)
                result.save(out_file)
                logging.info(f'掩码已保存: {out_file}')

            # 可视化结果
            if args.viz:
                logging.info(f'显示结果: {in_file} (关闭窗口继续)')
                plot_img_and_mask(img, mask_pred)

        except Exception as e:
            logging.error(f"处理图像 {in_file} 时出错: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()