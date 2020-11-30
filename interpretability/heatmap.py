import torch
from torch import nn
# CAM
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation

import cv2
import numpy as np
from skimage import io
import os

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = np.float32(heatmap) + np.float32(image)
    return norm_image(cam), heatmap.astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    计算guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().squeeze().detach().numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def save_image(image_dicts, input_image_name, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        cv2.imwrite(os.path.join(output_dir, '{}-{}.jpg'.format(prefix, key)), image)


def heat_map(image, tensors, model, class_id=None, layer_name=None, save_path='./heatmap/'):
    """
    生成热值图
    :param image: 图片原始numpy对象
    :param tensors: 图像tensor对象（之所以单独读取，是因为数据增强方式会变化）
    :param model: 模型对象
    :param class_id: 识别类别索引，为None时去概率最大的类别
    :param layer_name: 可视化的卷积层名称，请参考get_conv_name读取，默认最后一层
    :param save_path: 结果保存路径，默认'./heatmap/'
    :return:
    """
    image_dict = {}
    if layer_name is None:
        layer_name = get_last_conv_name(model)
    grad_cam = GradCAM(model, layer_name)
    mask = grad_cam(tensors, class_id)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(image, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(model, layer_name)
    mask_plus_plus = grad_cam_plus_plus(tensors, class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(image, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(model)
    # image_wakan.zero_grad()  # 梯度置零
    grad = gbp(tensors)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_image(image_dict, 'heatmap', save_path)


