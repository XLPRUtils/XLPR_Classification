# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
import os


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=Variable(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1])), gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = f.softmax(inputs, 1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class Criterion(nn.Module):
    def __init__(self, class_number=3, lambda_list=[1.0, 1.0, 1.0]):
        super(Criterion, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
        self.MSELoss = torch.nn.MSELoss(reduction='mean')
        self.BCELoss = torch.nn.BCELoss(weight=torch.from_numpy(np.array(lambda_list)).float().cuda(),reduction='mean')
        self.lambda_list = torch.Tensor(lambda_list)
        self.class_number = class_number
        self.focal_loss = FocalLoss(class_num=self.class_number)

    def class_balance_loss(self, out, label):
        beta = 0.9999
        gamma = 2.0
        samples_per_cls = [1, 1, 1, 1, 1]
        loss_type = "focal"
        cb_loss = CB_loss(label, out, samples_per_cls, self.class_number, loss_type, beta, gamma)

        return cb_loss
    # 计算交叉熵loss
    def cross_entropy_loss(self, out, label):

        return self.CrossEntropyLoss(out, label)

    def ohem_loss(self, cls_pred, cls_target):
        batch_size = cls_pred.size(0)
        ohem_cls_loss = f.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * 0.5))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss

    # 计算MSE_loss
    def mse_loss(self, out, label):
        out = out.argmax(dim=1).float().requires_grad_(requires_grad=True)
        # print(out, label)
        label = label.float().requires_grad_(requires_grad=True)
        return self.MSELoss(out, label)

    # 计算BCE_loss
    def bce_loss(self, out, label):
        one_hot = torch.nn.functional.one_hot(label, num_classes=self.class_number).float().cuda()
        out =  torch.sigmoid(out).requires_grad_(requires_grad=True)
        return self.BCELoss(out, one_hot)

    # 总loss,请在此处天马行空,例子是交叉熵,MSE,BCE的组合形势,可以自己去修改
    def forward(self, out, label):

        #   组合loss 尝试探索模块，后面如果要使用到多种loss的组合可以开启这里
        # loss_list = torch.Tensor([self.cross_entropy_loss(out, label), self.mse_loss(out, label),
        #                           self.bce_loss(out, label)]).requires_grad_(requires_grad=True)
        # loss_list包括了上述的三种loss，与相应的lambda值进行对位相乘
        # total_loss = loss_list.mul(self.lambda_list).sum()

        #   单一loss,交叉熵
        #   特别提醒,考虑到了样本不均衡问题,我在loss里面加了类别权重,如果类别数发生改变记得修改
        # total_loss = self.focal_loss(out, label)
        focal_loss = self.focal_loss(out, label)
        ohem_loss = self.ohem_loss(out, label)
        cs_loss = self.cross_entropy_loss(out, label)
        # cb_loss = self.class_balance_loss(out, label)

        # print(focal_loss , ohem_loss, cs_loss)
        total_loss = ohem_loss + cs_loss + focal_loss
        # total_loss = cs_loss + cb_loss
        # total_loss = self.cross_entropy_loss(out, label)
        return total_loss

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if __name__ == '__main__':
#     criterion = Criterion()
#     # # # 测试网络格式
#     # # # print(model)
#     # # # test_data = torch.randn((2, 3, 384, 768)).cuda()
#     # # # print(format(test_data.size()))
#     # # # test_out = model(test_data).cuda()
#     test_out = torch.FloatTensor([[3.1, 0.4, 0.7, 0.6, 9.7], [3.5, 0.8, 0.7, 0.6, 9.7], [9.9, 4.6, 0.7, 0.6, 9.7],[10.4, -0.4, 0.7, 0.6, 9.7]])
#     # # # print(format(test_out.size()[0]))
#     test_label = torch.LongTensor([4, 4, 4, 4])
#     # # # print(format(test_label.size()))
#     # # loss_0 = criterion.cs_score(0, test_out, test_label)
#     # # loss_1 = criterion.cs_score(1, test_out, test_label)
#     loss = criterion(test_out, test_label)
#     print(loss)

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = f.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = f.one_hot(labels, no_of_classes).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.tensor(weights).float().to(device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = f.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = f.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

if __name__ == '__main__':
    no_of_classes = 6
    logits = torch.rand(10,no_of_classes).float()
    labels = torch.randint(0,no_of_classes, size = (10,))
    print(logits, labels)
    loss = Criterion(class_number=no_of_classes)
    print(loss(logits, labels))

