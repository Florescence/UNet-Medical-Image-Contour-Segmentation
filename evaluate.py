import os

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils.dice_score import multiclass_dice_coeff, dice_coeff

import matplotlib.pyplot as plt

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, epoch_pred_dir):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        batch_index = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # # 打印真实掩码的尺寸
            # print(f"真实掩码尺寸: {mask_true.shape}")

            # predict the mask
            mask_pred = net(image)

            # 处理预测掩码的维度（关键修改）
            # if mask_pred.dim() == 4 and mask_pred.shape[1] == 1:
            #     mask_pred = mask_pred.squeeze(1)

            # # 打印预测掩码的尺寸
            # print(f"预测掩码尺寸: {mask_pred.shape}")

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                # mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # 1. 将logits转换为概率
                mask_pred = torch.sigmoid(mask_pred.squeeze(1))
                # 2. 阈值化转换为0/1标签（默认阈值0.5）
                mask_pred = (mask_pred > 0.5).float()
                plt.imshow(mask_pred[0].cpu().numpy(), cmap='gray')
                # 3. 此时mask_pred形状为[B, H, W]，值为0或1
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                batch_index += 1

                for i in range(len(mask_pred)):
                    pred_np = mask_pred[i].cpu().numpy()
                    save_path = os.path.join(epoch_pred_dir, f'pred_batch{batch_index}_sample{i}.png')
                    # 转换为0-255的uint8格式
                    pred_png = (pred_np * 255).astype(np.uint8)
                    Image.fromarray(pred_png).save(save_path)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
