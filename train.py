# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.distributions import normal
import torchvision.models as models
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm # 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器
import os
import argparse
from torchvision import datasets
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from Utils import utils, Criterion, Optimizer, Scheduler, Trainer
from apex import amp
from Networks.Model import Model
from Dataloaders.Dataloader import get_data_set
from Dataloaders import Dataloader
from torch.utils.tensorboard import SummaryWriter

# 这里并不是很推荐这么做，可以注释下面这一行
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
# 数据集相关
parser.add_argument('--data_set', type=str, default='gesture', choices=['gesture'],help='Default gesture')

# 模型相关
parser.add_argument('--model_type', type=str, default='Resnet34',
                    choices=['Resnet34', 'Resnet50', 'Resnet101',
                             'Resnet_CBAM34', 'Resnet_CBAM50', 'Resnet_CBAM101',
                             'Resnext34', 'Resnext50', 'Resnext101',
                             'Res2net50', 'Res2net101',
                             'SEnet34', 'SEnet50', 'SEnet101',
                             'EfficientnetB0', 'EfficientnetPre'],
                    help='The Net will be trained.Default EfficientnetB0')


# 训练相关的超参数
parser.add_argument('--max_epochs', type=int, default=60, help='The max number of train.Default 10')
parser.add_argument('--optimizer', type=str, default='Adam',
                    choices=['SGD', 'ASGD', 'Adam', 'Rprop', 'Adagrad',
                             'Adadelta', 'RMSprop', 'Adamax', 'SparseAdam', 'LBFGS'],
                    help='Default gesture')
parser.add_argument('--lr_scheduler', type=str, default='WarmupPolyLR',
                    choices=['MultiStepLR', 'StepLR', 'ExponentialLR', 'WarmupPolyLR'],
                    help='Default gesture')
parser.add_argument('--seed', type=int, default=1024, help='Default 1024')
parser.add_argument('--batch_size', type=int, default=144, help='The batch_size of Dataloader.Defalut 64')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight_decay,Default 1e-3')
parser.add_argument('--load_epoch', type=int, default=-1, help='The epoch that will load model.Default -1')
parser.add_argument('--save_epoch', type=int, default=20, help='How many epochs save a model.Default 1')
parser.add_argument('--apex', default=True, help='Turn on the mixed accuracy training Apex')

# 显示相关，使用nohup挂后台运行时请关闭tqdm
parser.add_argument('--tqdm_off', action='store_true', default=False, help='Whether turn of the progress bar.Default False')
parser.add_argument('--details', default=True, help='Detailed category identification scores')
parser.add_argument('--tensorboardX', default=False, help='Open tensorboardX for visualization of model, Loss and Accuracy')


args = parser.parse_args()


if __name__ == '__main__':
    save_path = 'results/'

    # 不存在文件夹则创建
    if not os.path.exists(save_path):
        print('savepath is not exit, create {}'.format(save_path))
        os.makedirs(save_path)

    # 随机种子，保持一致性，保证每次初始化随机数一致
    torch.manual_seed(args.seed)
    # 如果使用多个GPU为所有的GPU设置种子。
    torch.cuda.manual_seed_all(args.seed)

    # 训练的时候打印训练日志

    logger_path = os.path.join(save_path, 'logger')
    # 创建日志文件夹
    if not os.path.exists(logger_path):
        print('logger_path is not exit, create {}'.format(logger_path))
        os.makedirs(logger_path)
    train_logger = utils.TextLogger('train : ', '{}/train_{}_{}.log'.format(logger_path, args.data_set, args.model_type))

    # 进度条
    if args.tqdm_off:
        def nop(it, *a, **k):
            return it
        tqdm = nop

    
    # 加载数据  train默认是培训数据前0.8   val是默认是培训数据后0.2 , test是测试集
    # 可能不存在val集，直接train+test, 依照数据集情况来定
    train_data_set = get_data_set(type=args.data_set, split='train')
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=args.batch_size, shuffle=True, num_workers=10)
    print('train dataset len: {}'.format(len(train_data_loader.dataset)))

    val_data_set = get_data_set(type=args.data_set, split='val')
    val_data_loader = torch.utils.data.DataLoader(val_data_set,
                                                       batch_size=args.batch_size, num_workers=10)
    print('val dataset len: {}'.format(len(val_data_loader.dataset)))
    print('load dateset finished')

    # 初始化模型
    model = Model(args.model_type, num_classes=train_data_set.CLASS)
    # 将模型的结构写入tensorboardX
    if args.tensorboardX:
        writer = SummaryWriter('{}/train_{}_{}'.format(logger_path, args.data_set, args.model_type))
        writer.add_graph(model, torch.rand([1, 3, 64, 64]))

    # 定义优化器
    opt = Optimizer.get_optimizer(type=args.optimizer,model=model, lr=args.lr, wd=args.wd)
    # 定义学习策略
    sch = Scheduler.get_lr_scheduler(type=args.lr_scheduler, opt=opt, max_epoch=args.max_epochs, iter_epoch=len(train_data_loader))
    # 损失函数，交叉熵
    ent_loss = Criterion.Criterion(class_number=train_data_set.CLASS)
    # 模型和损失函数都放上cuda
    if torch.cuda.is_available():
        model.cuda()
        ent_loss = ent_loss.cuda()


    # 恢复上次训练的位置, 继续训练，根据load_epoch的数值手动加载上一次的训练结果
    epoch = 0
    if args.load_epoch != -1:
        epoch = args.load_epoch
        model.load_state_dict(torch.load('%s/%s_%s_%d.pth' % (save_path, args.data_set, args.model_type, epoch)))

    # Apex初始化
    if args.apex:
        model, opt = amp.initialize(model, opt, opt_level="O1")

    # 多卡并行模型初始化,用不着那么多显存
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()  # CUDA_VISIBLE_DEVICES=5,6设置环境可见,自动适应显卡数量

    # 训练
    best_Score_f1 = 0
    print('training')
    while True:
        # print("学习率：" + str(sch.get_lr()[0]))
        # print("学习率：" + str(opt.param_groups[0]['lr']))
        # 训练集训练
        train_f1_score_, train_accuracy_score_, loss = Trainer.train(model, train_data_loader, opt, ent_loss, epoch, train_logger, tqdm, args.apex, args.details, train_data_set.CLASS)

        # 动态改变学习率，根据epoch判断当前的学习率是否需要改变
        sch.step()
        # 验证集验证
        val_f1_score_, val_accuracy_score_, _ = Trainer.val(model, val_data_loader, epoch, train_logger, tqdm, args.details, train_data_set.CLASS)

        # 向TensorBoardX中写入log信息，这里以Acc和Loss为例
        if args.tensorboardX:
            writer.add_scalar('train_accuracy', train_accuracy_score_, epoch)
            writer.add_scalar('val_accuracy', val_accuracy_score_, epoch)
            writer.add_scalar('train_loss', loss, epoch)

        # 保存模型快照
        if epoch % args.save_epoch == 0:
            print('save model')
            torch.save(model.module.state_dict(), '%s/%s_%s_%d.pth' % (save_path, args.data_set, args.model_type, epoch))

        # 随时保存test集上最高的F1模型
        if val_f1_score_ > best_Score_f1:
            best_Score_f1 = val_f1_score_
            print('save the best model')
            torch.save(model.module.state_dict(), '%s/%s_%s_best.pth' % (save_path, args.data_set, args.model_type))

        # 到达最大迭代epoch，退出训练
        if epoch == args.max_epochs:
            break

        epoch += 1
