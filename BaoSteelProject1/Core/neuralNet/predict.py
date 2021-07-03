#!/usr/bin/env python
# coding:utf-8
"""
@Name : predict.py
@Author  : May
@Email : 3237351801@qq.com
@Time    : 12/26/20 1:34 PM
"""

# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:53
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : inference.py
"""

"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from Core.neuralNet.NET.model import NestedUNet
from Core.neuralNet.NET.model import UNet
# from NET.model import NestedUNet
# from NET.model import UNet
from Core.neuralNet.utils.dataset import BasicDataset
from Core.neuralNet.config import NetConfig


class Predict():
    def __init__(self, chkpth, n_classes):
        self.chkpth = chkpth
        self.cfg = NetConfig(model="UNet",n_classes=n_classes)
        self.net = eval(self.cfg.model)(self.cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=self.device)
        self.net.load_state_dict(torch.load(self.chkpth,map_location=self.device))

    def inference_one(self, net, image, device):
        net.eval()

        img = torch.from_numpy(BasicDataset.preprocess(image, self.cfg.scale))

        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = net(img)
            if self.cfg.deepsupervision:
                output = output[-1]

            if self.cfg.n_classes > 1:
                probs = F.softmax(output, dim=1)
            else:
                probs = torch.sigmoid(output)

            probs = probs.squeeze(0)

            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((image.shape[1], image.shape[0])),
                    transforms.ToTensor()
                ]
            )

            if self.cfg.n_classes == 1:
                probs = tf(probs.cpu())
                mask = probs.squeeze().cpu().numpy()
                return mask > self.cfg.out_threshold
            else:
                masks = []
                for prob in probs:
                    prob = tf(prob.cpu())
                    mask = prob.squeeze().cpu().numpy()
                    mask = mask > self.cfg.out_threshold
                    masks.append(mask)
                return masks

    def predict(self,image, stepLength):
        height, width = image.shape[0], image.shape[1]
        Mask = np.zeros((height,width))
        for i in range(stepLength, height + stepLength,stepLength):
            for j in range(stepLength, width + stepLength, stepLength):
                if i > height:
                    i = height
                if j > width:
                    j = width
                img = image[i-stepLength:i,j-stepLength:j]
                subMask = self.inference_one(net= self.net, image = img, device = self.device)
                small_mask = np.zeros((stepLength, stepLength))
                for idx in range(0, len(subMask)):
                    small_mask[subMask[idx] == 1] = idx
                Mask[i-stepLength:i,j-stepLength:j] = small_mask
        return Mask


