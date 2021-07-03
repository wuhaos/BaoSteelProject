#!/usr/bin/env python
# coding:utf-8
"""
@Name : dataset.py
@Author  : May
@Email : 3237351801@qq.com
@Time    : 12/23/20 4:32 PM
"""
import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2 as cv


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, img, scale):
        # w, h = img.shape[1], img.shape[0]
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newH, newW))
        #
        # img_nd = np.array(pil_img)

        # if len(img_nd.shape) == 2:
        #     # mask target image
        #     img_nd = np.expand_dims(img_nd, axis=2)
        # else:
        #     # grayscale input image
        #     # scale between 0 and 1
        #     img_nd = img_nd / 255
        img = img/255
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        # img_path = osp.join(self.imgs_dir, img_name)
        # mask_path = osp.join(self.masks_dir, img_name)
        img_path = self.imgs_dir+img_name
        mask_path = self.masks_dir+img_name

        img = cv.imread(img_path)
        mask = cv.imread(mask_path,0)

        # mask = np.zeros((512, 512, 8))
        # """convert gray_label_img to eight channels mask"""
        # for i in range(8):
        #     mask[mask_tmp == i, i] = 1
        # assert img.size == mask_tmp.size, \
        #     f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
