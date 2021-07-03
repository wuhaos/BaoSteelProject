# #!/usr/bin/env python
# # coding:utf-8
# """
# @Name : predict.py
# @Author  : May
# @Email : 3237351801@qq.com
# @Time    : 12/26/20 1:34 PM
# """
#
# # -*- coding: utf-8 -*-
# # @Time    : 2020-02-26 17:53
# # @Author  : Zonas
# # @Email   : zonas.wang@gmail.com
# # @File    : inference.py
# """
#
# """
# import argparse
# import logging
# import os
# import os.path as osp
# import cv2 as cv
#
#
# EnumColor = {
#     0: (0, 0, 0),
#     1: (255, 0, 255),
#     2: (128, 128, 128),
#     3: (0, 0, 255),
#     4: (0, 255, 255),
#     5: (255, 255, 255),
#     6: (0, 0, 128),
#     7: (0, 128, 128)
#
# }
#
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# testPath = '/home/may/Documents/BaoSteel/Step1/Data/PreRawDataSet/'
# testMaskPath = '/home/may/Documents/BaoSteel/Step1/Data/RawTestMask_v2/'
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
#
# from NET.model import NestedUNet
# from NET.model import UNet
# from utils.dataset import BasicDataset
# from config import NetConfig
#
# cfg = NetConfig(model='UNet')
#
#
#
#
# def inference_one(net, image, device):
#     net.eval()
#
#     img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))
#
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)
#
#     with torch.no_grad():
#         output = net(img)
#         if cfg.deepsupervision:
#             output = output[-1]
#
#         if cfg.n_classes > 1:
#             probs = F.softmax(output, dim=1)
#         else:
#             probs = torch.sigmoid(output)
#
#         probs = probs.squeeze(0)
#
#         tf = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.Resize((image.shape[1], image.shape[0])),
#                 transforms.ToTensor()
#             ]
#         )
#
#         if cfg.n_classes == 1:
#             probs = tf(probs.cpu())
#             mask = probs.squeeze().cpu().numpy()
#             return mask > cfg.out_threshold
#         else:
#             masks = []
#             for prob in probs:
#                 prob = tf(prob.cpu())
#                 mask = prob.squeeze().cpu().numpy()
#                 mask = mask > cfg.out_threshold
#                 masks.append(mask)
#             return masks
#
#
# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--model', '-m', default='MODEL.pth',
#                         metavar='FILE',
#                         help="Specify the file in which the model is stored")
#     parser.add_argument('--input', '-i', dest='input', type=str, default='',
#                         help='Directory of input images')
#     parser.add_argument('--output', '-o', dest='output', type=str, default='',
#                         help='Directory of ouput images')
#     return parser.parse_args()
#
#
# if __name__ == "__main__":
#     args = get_args()
#
#     # input_imgs = os.listdir(args.input)
#
#     net = eval(cfg.model)(cfg)
#     # net = eval(Configure.model)(Configure)
#     logging.info("Loading model {}".format(args.model))
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # logging.info(f'''
#     #     Using device: {device.type}
#     # ''')
#     net.to(device=device)
#     # net.load_state_dict(torch.load(args.model, map_location=device))
#     net.load_state_dict(torch.load('/home/may/Documents/BaoSteel/Step1/v2/Data/CrossEntropy_UNet/UNet_lr_e_3__weight_decay_e_4___epoch_31.pth', map_location=device))
#
#     logging.info("Model loaded !")
#
#
#     for list in os.listdir(testPath):
#         bigImg = cv.imread(testPath+list)
#         height, width = bigImg.shape[0], bigImg.shape[1]
#         bigIdx = np.zeros((height,width))
#         bigMask = np.zeros((height,width,3))
#         for i in range(800,height+800,800):
#             for j in range(800,width+800,800):
#                 if i > height:
#                     i = height
#                 if j > width:
#                     j = width
#                 img = bigImg[i-800:i,j-800:j]
#                 small_mask = np.zeros((800,800))
#                 mask = inference_one(net=net,
#                                      image=img,
#                                      device=device)
#
#                 # print("mask size: ", len(mask))
#
#
#                 for idx in range(0, len(mask)):
#                     small_mask[mask[idx] == 1] = idx
#                 bigIdx[i-800:i,j-800:j] = small_mask
#         bigMask[bigIdx==0] = EnumColor[0]
#         bigMask[bigIdx==1] = EnumColor[1]
#         bigMask[bigIdx==2] = EnumColor[2]
#         bigMask[bigIdx==3] = EnumColor[3]
#         bigMask[bigIdx==4] = EnumColor[4]
#         bigMask[bigIdx==5] = EnumColor[5]
#         bigMask[bigIdx==6] = EnumColor[6]
#         bigMask[bigIdx==7] = EnumColor[7]
#         cv.imwrite(testMaskPath+list, bigMask)
#
#
#
#

import calendar
import datetime


now = datetime.datetime.now().strftime("%Y.%m.%d %H:%M")
print(now)
