#!/usr/bin/env python
# coding:utf-8
"""
@Name : config.py
@Author  : May
@Email : 3237351801@qq.com
@Time    : 12/23/20 4:22 PM
"""
import os


class NetConfig:

    def __init__(self,
                 epochs = 300,  # Number of epochs
                 batch_size = 1,    # Batch size
                 validation = 10.0,   # Percent of the data that is used as validation (0-100)
                 out_threshold = 0.5,

                 optimizer='SGD',
                 # lr = 0.0001,     # learning rate
                 lr=0.00005,  # learning rate
                 lr_decay_milestones = [20, 50],
                 lr_decay_gamma = 0.9,
                 # weight_decay=1e-8,
                 weight_decay=1e-4,
                 momentum=0.9,
                 nesterov=True,

                 in_channels = 3, # Number of channels in input images
                 n_classes = 8,  # Number of classes in the segmentation
                 scale = 1,    # Downscaling factor of the images

                 load = False,   # Load model from a .pth file
                 save_cp = True,

                 model='NestedUNet',
                 bilinear = True,
                 deepsupervision = False,
                 ):
        """

        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param validation: Percent of the data that is used as validation (0-100)
        :param out_threshold:
        :param optimizer:
        :param lr: learning rate
        :param lr_decay_milestones:
        :param lr_decay_gamma:
        :param weight_decay:
        :param momentum:
        :param nesterov:
        :param in_channels: Number of channels in input images
        :param n_classes: Number of classes in the segmentation
        :param scale: Downscaling factor of the images
        :param load: Load model from a .pth file
        :param save_cp:
        :param model:
        :param bilinear:
        :param deepsupervision:
        """
        super(NetConfig, self).__init__()

        self.images_dir = '/home/may/Documents/BaoSteel/Step1/v2/Data/Origin_1/'
        self.masks_dir = '/home/may/Documents/BaoSteel/Step1/v2/Data/Mask_1/'
        self.checkpoints_dir = '/home/may/Documents/BaoSteel/Step1/v2/Data/CrossEntropy_'+model+'/'

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation = validation
        self.out_threshold = out_threshold

        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay_milestones = lr_decay_milestones
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.scale = scale

        self.load = load
        self.save_cp = save_cp

        self.model = model
        self.bilinear = bilinear
        self.deepsupervision = deepsupervision

        os.makedirs(self.checkpoints_dir, exist_ok=True)
