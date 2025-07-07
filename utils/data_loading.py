import logging
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_mask', augment=True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.augment = augment

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

    def __len__(self):
        # 若启用增强，数据集大小变为原来的4倍（原始+3种旋转）
        return len(self.ids) * (4 if self.augment else 1)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int8)
            mask[img == 255] = 2  # 目标轮廓
            mask[img == 128] = 1  # 背景
            mask[img == 0] = 0  # 虚影
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img.astype(np.float32) / 255.0

            return img

    @staticmethod
    def rotate_image_and_mask(img, mask, angle):
        """旋转图像和掩码"""
        # 旋转图像
        rotated_img = img.rotate(angle, expand=True)
        # 旋转掩码（保持标签值不变）
        rotated_mask = mask.rotate(angle, expand=True)
        return rotated_img, rotated_mask

    def __getitem__(self, idx):
        # 计算原始索引和旋转偏移
        original_idx = idx // 4 if self.augment else idx
        rotation_idx = idx % 4 if self.augment else 0

        name = self.ids[original_idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # 应用旋转增强
        if self.augment and rotation_idx > 0:
            angles = [90, 180, 270]
            img, mask = self.rotate_image_and_mask(img, mask, angles[rotation_idx - 1])

        # 预处理
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        assert np.all(np.logical_or.reduce([mask == v for v in [0, 1, 2]])), "掩码包含非法类别索引"

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, augment=True):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', augment=augment)