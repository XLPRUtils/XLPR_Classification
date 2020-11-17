import cv2
from PIL import Image
import os
import glob
import numpy as np
import math
import re
import torch
from torch.utils import data
from torchvision import transforms
from Networks.Model import Model
from Dataloaders.Dataloader import get_data_set
from Dataloaders import Dataloader

if __name__ == '__main__':
    # 初始化数据集，加载测试集
    test_data_set = get_data_set(type='gesture', split='test')
    test_data_loader = torch.utils.data.DataLoader(test_data_set,
                                                    batch_size=16, shuffle=False, num_workers=10)
    print('train dataset len: {}'.format(len(test_data_loader.dataset)))

    # 初始化模型
    model = Model('Resnet34', num_classes=test_data_set.CLASS)
    model.load_state_dict(torch.load("./results/gesture_Resnet34_best.pth"))
    model.cuda().eval()

    # # 批量分析
    # for i, (data, target) in enumerate(test_data_loader):
    #     if torch.cuda.is_available():
    #         data = data.cuda()
    #     with torch.no_grad():
    #         pre_cls = model(data)
    #     classification_result = torch.argmax(pre_cls, 1).cpu().detach().numpy()
    #     print(classification_result, target.detach().numpy())

    # 数据预处理初始化
    normal_data = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = cv2.imread('./test/wakan.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

    image_wakan = normal_data(image).unsqueeze(0)
    if torch.cuda.is_available():
        image_wakan = image_wakan.cuda()
    with torch.no_grad():
        pre_cls = model(image_wakan)
    classification_result = torch.argmax(pre_cls, 0).cpu().detach().numpy()
    print('wakan image inference result is: {}'.format(classification_result))
