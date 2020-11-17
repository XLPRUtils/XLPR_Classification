# -*- coding:utf-8 -*-
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import albumentations
import cv2

# 数据增强包
from albumentations import (
   Compose,Resize,OneOf,RandomBrightness,RandomContrast,MotionBlur,MedianBlur,
   GaussianBlur,VerticalFlip,HorizontalFlip,ShiftScaleRotate,Normalize,
)


class GestureDataSet(data.Dataset):

    def __init__(self, dataset_path='/home/hly/GestureData/', split='train'):

        # 初始化参数，可以根据情况修改
        self.root_path = dataset_path # 数据集路径
        self.size = [64, 64] # resize尺寸,[h,w]
        self.split = split # 选取什么集[train/val/test]
        self.transform = False # 是否进行数据增强，默认只有训练集增强
        self.train_ratio = 0.8 # 培训集拆分训练集、测试比例
        self.CLASS = 6 # 类别数

        # 根据split读取数据集对应txt索引文件
        if split == 'test':
            self.data = open(os.path.join(self.root_path, 'test.txt'), 'r').readlines()

        else:
            self.data = open(os.path.join(self.root_path, 'train.txt'), 'r').readlines()
            # 根据self.train_ratio切分训练集和验证集，如果是训练集开启数据增强
            if split == 'train':
                self.data = self.data[:int(len(self.data)*self.train_ratio)]
                self.transform = True
            else:
                self.data = self.data[int(len(self.data) * self.train_ratio):]

        # pytorch数据转换器，zero-标准化默认使用VOC的均值和方差，也可以通过mean_std.py计算数据集的均值和方差并替换
        # 若数据集分布本身极不符合类正态分布，加入标准会则会降低精度
        self.transforms_data = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
    # 数据列表存入self.data中
    def __getitem__(self, index):
        # 根据index索引值读取对应数据的图片路径和标签
        data_path, target = os.path.join(self.root_path, self.data[index][9:-1]).split(' ')
        # 读取图片并resize尺寸
        image = cv2.imread(data_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        # 如果是训练集进行数据增强，详细增强可以查看albumentations_transform
        if self.transform:
            image = self.albumentations_transform(image)
        image = self.transforms_data(image)
        # 标准格式转换
        target = int(target)

        # 返回单个数据的数据和标签，type(tensor)
        return image, target

    def __len__(self):
        # 定义数据集的大小
        return len(self.data)

    def data_distribution(self):
        number = [0] * self.CLASS
        for k in self.data:
            type = int(k[-2:-1])
            number[type] += 1

        labels = [str(i) for i in range(self.CLASS)]
        # 画图
        plt.pie(x=number, labels=labels)
        # 展示
        plt.show()



    # 数据增强函数
    def albumentations_transform(self, img):
        # 训练集的增强
        trans_train = Compose([
            # 随机更改输入图像的色相，饱和度和值
            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
            # 通过用255减去像素值来反转输入图像。
            # InvertImg(always_apply=False, p=1),
            # 随机改变RGB三个通道的顺序
            # ChannelShuffle(always_apply=False, p=0.5),
            # 随机出现小黑点
            # Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            # RandomCrop(224, 224),
            OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
            # OneOf([MotionBlur(blur_limit=3),MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3),], p=0.5,),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            # ShiftScaleRotate(
            #     shift_limit=0.2,
            #     scale_limit=0.2,
            #     rotate_limit=20,
            #     interpolation=cv2.INTER_LINEAR,
            #     border_mode=cv2.BORDER_REFLECT_101,
            #     p=1,
            # ),
        ])

        augmented = trans_train(image=img)
        return augmented['image']



if __name__ == '__main__':

    dataset = GestureDataSet(split='train')
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=10)
    # 输出DataSet的大小，以及DataLoader的大小
    print(len(dataset),train_data_loader.__len__())

    # # 可视化图像
    # for _, (data, targets) in enumerate(train_data_loader):
    #     print(_, data.shape, targets.shape, targets)
    #     fig = plt.figure()
    #     for i in range(16):
    #         ax = fig.add_subplot(4, 4, i+1)
    #         plt.axis('off')  # 去掉每个子图的坐标轴
    #         plt.imshow(data[i].permute(1,2,0))
    #     plt.subplots_adjust(wspace=0, hspace=0)  # 修改子图之间的间隔
    #     plt.show()
    #     break

    # 数据分布统计
    dataset.data_distribution()



