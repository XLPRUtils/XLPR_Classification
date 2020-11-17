#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
from Dataloaders import Dataloader
import torchvision.models as models

import os
import sys
import math

import numpy as np

dataset = Dataloader.__dict__['MoticClassificationDataSet']
data_root = '/home/hly/Data/Bladder_competition/'
save_root = './save/'
num_workers = 1
cancer = 2

batch_size_train = 310
batch_size_test = 80
max_epoch = 800
# criterion = nn.MSELoss()

train_data_loader = torch.utils.data.DataLoader(dataset(dataset_path='/home/dataset/Motic/', transform=True, split='train'),
                                                batch_size=100, shuffle=True, num_workers=5, drop_last=False)
print('train dataset len: {}'.format(len(train_data_loader.dataset)))

Mean = np.array([0.,0.,0.]).astype(np.float32)
Std = np.array([0.,0.,0.]).astype(np.float32)
# # 输出数据格式
for batch_datas, batch_labels in train_data_loader:
    # print(batch_datas.size(),batch_datas.type())
    # print(batch_labels.size(), batch_labels.type())
    batch_datas_np = batch_datas.numpy()
    batch_datas_np = batch_datas_np.astype(np.float32)
    means = []
    stdevs = []
    for i in range(3):
        pixels = batch_datas_np[:, i, :, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    Mean+=means
    Std+=stdevs

    # print("means: {}".format(Mean))
    # print("stdevs: {}".format(Std))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
print('transforms.Normalize(Mean = {}, Std = {})'.format(Mean/len(train_data_loader), Std/len(train_data_loader)))

    # label = batch_labels
    # label = label.float()
    # print(batch_labels)
    # print(batch_datas)
    # print(data.size)
    # break