import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet_S, UNet
from unet.unet_model import UNet_SA, UNet_T
from unet.unet_nested_model import UNetPlusPlus_S, UNetPlusPlus
from utils.boundary_loss import boundary_loss
from yolo.yolov8_seg_model import YOLOv8_Seg_S
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

data_root = Path('data/data-without-black-shadow')
dir_checkpoint = Path('./checkpoints/')
dir_img_train = data_root / 'imgs/train'
dir_img_val = data_root / 'imgs/val'
dir_mask_train = data_root / 'masks/train'
dir_mask_val = data_root / 'masks/val'

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. 创建数据集
    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    train_set = BasicDataset(dir_img_train, dir_mask_train, img_scale)
    val_set = BasicDataset(dir_img_val, dir_mask_val, img_scale)

    # 2. 分割训练/验证集
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # n_val = len(val_set)
    n_train = len(train_set)

    # 3. 创建数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # # 初始化logging
    # experiment = wandb.init(project='U-Net', resume="auto", mode="offline")
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {train_set.__len__() / (train_set.__len__() + val_set.__len__())}
        Validation size: {val_set.__len__() / (train_set.__len__() + val_set.__len__())}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. 设置optimizer, scheduler, loss以及loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2, eta_min=1e-7)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        # 创建当前epoch的预测结果保存目录，图片保存逻辑在evaluate.py
        # save_pred = epoch % 5 == 0  # 每5个epoch保存一次预测结果
        # if save_pred:
        #     epoch_pred_dir = Path(f'./predictions/epoch_{epoch}')
        #     epoch_pred_dir.mkdir(parents=True, exist_ok=True)
        # else:
        #     epoch_pred_dir = None

        epoch_pred_dir = Path(f'./predictions/epoch_{epoch}')
        epoch_pred_dir.mkdir(parents=True, exist_ok=True)

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1: # 二分类，对于清晰图像使用，仅区分前景人体和背景非人体
                        true_masks //= 2 # 导入时2为前景1为背景，需要重置为1为前景0为背景
                        # 计算交叉熵损失
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        # 计算Dice损失
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # # 计算连通域惩罚损失
                        # pred_prob = torch.sigmoid(masks_pred.squeeze(1))
                        # cc_loss = connected_component_loss(
                        #     pred_prob,
                        #     edge_distance=50,
                        #     min_area=1000,
                        #     penalty_weight=0.1
                        # )
                        # loss += cc_loss
                        # 计算边界损失
                        loss += 0.25 * boundary_loss(masks_pred.squeeze(1), true_masks.float(), edge_width=51, edge_weight=15)

                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        # if epoch > epochs / 2:
                        #     progress = min(1.0, (epoch - 25) / 10.0)
                        #     boundary_weight = progress * 1.0
                        #     norm_factor = 0.2
                        #     loss += boundary_weight * norm_factor * boundary_loss(masks_pred, true_masks.float(), edge_width=51, edge_weight=7)

                if torch.isnan(loss).any():
                    # torch.save(model.state_dict(), 'model_before_nan.pth')
                    raise RuntimeError("Fatal: NaN loss detected!")

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })

                # pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.set_postfix(**{'loss (total)': epoch_loss})

                # 评估轮次
                # division_step = (n_train // (5 * batch_size))
                division_step = n_train // batch_size
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_score_postprocess, min_val_score = evaluate(model, val_loader, device, amp, epoch_pred_dir)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        logging.info('Validation Postprocessed Dice score: {}'.format(val_score_postprocess))
                        logging.info('Validation Min Dice score: {}'.format(min_val_score))
                        # try:
                        #     experiment.log({
                        #         'learning rate': optimizer.param_groups[0]['lr'],
                        #         'validation Dice': val_score,
                        #         'images': wandb.Image(images[0].cpu()),
                        #         'masks': {
                        #             'true': wandb.Image(true_masks[0].float().cpu()),
                        #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #         },
                        #         'step': global_step,
                        #         'epoch': epoch,
                        #         **histograms
                        #     })
                        # except:
                        #     pass

        if save_checkpoint:
            factor = 5 # 保存频率
            if epoch > epochs * 0.5: # 开始保存位置
                if epoch % factor == 0:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict = model.state_dict()
                    state_dict['mask_values'] = train_set.mask_values + val_set.mask_values
                    torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                    logging.info(f'Checkpoint {epoch} saved!')

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), f'model_epoch{epochs}.pth')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 修改此处以适配数据类型
    # 对于RGB图像设置n_channels=3
    # n_classes=每个像素点有几个概率值
    # # 微缩版UNet (1ms)
    # model = UNet_T(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # 轻量化UNet (5ms)
    model = UNet_S(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # # 标准Unet (40ms)
    # model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # # UNet_S + 空间注意力
    # model = UNet_SA(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    # # UNet++
    # model = UNetPlusPlus(n_channels=1, n_classes=1, bilinear=args.bilinear)
    # # Yolov8-seg-S，仅二分类 (5ms)
    # model = YOLOv8_Seg_S(n_channels=1, n_classes=1)
    model = model.to(memory_format=torch.channels_last)

    # # UNet
    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # YOLO
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
