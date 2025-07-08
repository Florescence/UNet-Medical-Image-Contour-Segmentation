import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.dice_score import dice_coeff
from utils.post_process import postprocess_mask


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, epoch_pred_dir=None, postprocess=True):
    """
    评估模型性能，支持原始预测和后处理后的评估

    Args:
        net: 模型
        dataloader: 数据加载器
        device: 计算设备
        amp: 是否使用混合精度
        epoch_pred_dir: 保存预测结果的目录
        postprocess: 是否应用后处理
        min_area: 后处理中最小连通域面积
        kernel_size: 后处理中形态学操作的核大小

    Returns:
        tuple: (原始预测Dice分数, 后处理后Dice分数, 最小Dice分数)
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score_original = 0
    dice_score_postprocessed = 0
    min_dice_score = 10

    # 创建后处理保存目录
    postprocessed_dir = None
    if epoch_pred_dir is not None and postprocess:
        postprocessed_dir = os.path.join(epoch_pred_dir, "postprocessed")
        os.makedirs(postprocessed_dir, exist_ok=True)

    # 遍历验证集
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        batch_index = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # 预测掩码
            mask_pred = net(image)

            if net.n_classes == 1:
                # 二分类处理
                mask_true //= 2
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

                # 1. 将logits转换为概率
                mask_pred_prob = torch.sigmoid(mask_pred.squeeze(1))
                # 2. 阈值化转换为0/1标签
                mask_pred_binary = (mask_pred_prob > 0.5).float()

                # 计算原始预测的Dice分数
                dice_original = dice_coeff(mask_pred_binary, mask_true, reduce_batch_first=False)
                dice_score_original += dice_original

                # 后处理
                if postprocess:
                    mask_pred_postprocessed = torch.zeros_like(mask_pred_binary)
                    for i in range(len(mask_pred_binary)):
                        # 转换为numpy数组进行后处理
                        pred_np = mask_pred_binary[i].cpu().numpy().astype(np.uint8) * 255
                        # 应用后处理
                        processed_np = postprocess_mask(pred_np)
                        # 转回torch tensor
                        processed_tensor = torch.from_numpy((processed_np // 255).astype(np.float32)).to(device)
                        mask_pred_postprocessed[i] = processed_tensor

                    # 计算后处理后的Dice分数
                    dice_postprocessed = dice_coeff(mask_pred_postprocessed, mask_true, reduce_batch_first=False)
                    dice_score_postprocessed += dice_postprocessed

                # 更新最小Dice分数
                current_dice = dice_original if not postprocess else min(dice_original, dice_postprocessed)
                if current_dice < min_dice_score:
                    min_dice_score = current_dice

                batch_index += 1

                # 保存预测掩码
                if epoch_pred_dir is not None:
                    for i in range(len(mask_pred_binary)):
                        # 保存原始预测
                        pred_np = mask_pred_binary[i].cpu().numpy()
                        save_path = os.path.join(epoch_pred_dir, f'pred_batch{batch_index}_sample{i}.png')
                        pred_png = (pred_np * 255).astype(np.uint8)
                        Image.fromarray(pred_png).save(save_path)

                        # 保存后处理预测
                        if postprocess:
                            post_np = mask_pred_postprocessed[i].cpu().numpy()
                            post_save_path = os.path.join(postprocessed_dir, f'pred_batch{batch_index}_sample{i}.png')
                            post_png = (post_np * 255).astype(np.uint8)
                            Image.fromarray(post_png).save(post_save_path)

            else:
                # 多分类处理
                mask_pred_indices = mask_pred.argmax(dim=1)  # [B, H, W]，值为0/1/2

                # 仅对目标class求dice
                c = 2
                pred_c = (mask_pred_indices == c).float()
                true_c = (mask_true == c).float()
                current_dice = dice_coeff(pred_c, true_c, reduce_batch_first=False)

                # 注意dice是按batch累加最后取平均
                dice_score_original += current_dice

                if current_dice < min_dice_score:
                    min_dice_score = current_dice

                # 后处理
                if postprocess:
                    mask_pred_postprocessed = torch.zeros_like(mask_pred_indices)
                    for i in range(len(mask_pred_indices)):
                        # 转换为numpy数组进行后处理
                        pred_np = mask_pred_indices[i].cpu().numpy().astype(np.uint8)
                        # 应用后处理
                        processed_np = postprocess_mask(pred_np)
                        # 转回torch tensor
                        processed_tensor = torch.from_numpy(processed_np).to(device)
                        mask_pred_postprocessed[i] = processed_tensor

                    # 计算后处理后的Dice分数
                    c = 2
                    pred_c = (mask_pred_postprocessed == c).float()
                    true_c = (mask_true == c).float()
                    dice_score_postprocessed += dice_coeff(pred_c, true_c, reduce_batch_first=False)

                batch_index += 1

                # 保存预测掩码
                if epoch_pred_dir is not None:
                    for i in range(len(mask_pred_indices)):
                        # 保存原始预测
                        pred_np = mask_pred_indices[i].cpu().numpy()
                        save_path = os.path.join(epoch_pred_dir, f'pred_batch{batch_index}_sample{i}.png')
                        pred_vis = np.zeros_like(pred_np, dtype=np.uint8)
                        pred_vis[pred_np == 0] = 0
                        pred_vis[pred_np == 1] = 128
                        pred_vis[pred_np == 2] = 255
                        Image.fromarray(pred_vis).save(save_path)

                        # 保存后处理预测
                        if postprocess:
                            post_np = mask_pred_postprocessed[i].cpu().numpy()
                            post_save_path = os.path.join(postprocessed_dir, f'pred_batch{batch_index}_sample{i}.png')
                            post_vis = np.zeros_like(post_np, dtype=np.uint8)
                            post_vis[post_np == 0] = 0
                            post_vis[post_np == 2] = 255
                            Image.fromarray(post_vis).save(post_save_path)

    net.train()

    # 如果没有进行后处理，后处理分数与原始分数相同
    if not postprocess:
        dice_score_postprocessed = dice_score_original

    return dice_score_original / max(num_val_batches, 1), dice_score_postprocessed / max(num_val_batches, 1), min_dice_score