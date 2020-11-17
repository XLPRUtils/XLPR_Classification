import torch
import numpy as np
import tqdm
from apex import amp
from sklearn.metrics import f1_score,precision_score,recall_score,cohen_kappa_score,accuracy_score
from prettytable import PrettyTable


# 计算f1值
def compute_weight_score(class_out, targets, details, number_class, avg_loss, train_logger, epoch):
    f1_score_ = f1_score(targets, class_out, average=None)
    precision_score_ = precision_score(targets, class_out, average=None)
    recall_score_ = recall_score(targets, class_out, average=None)

    accuracy_score_ = accuracy_score(targets, class_out)
    cohen_kappa_score_ = cohen_kappa_score(targets, class_out)
    show_list = ['F1', 'Precision', 'Recall', 'Acc', 'Kappa']
    score_list = [np.mean(f1_score_), np.mean(precision_score_), np.mean(recall_score_), accuracy_score_, cohen_kappa_score_]
    if avg_loss is not None:
        show_list.append('Loss')
        score_list.append(avg_loss)
        hand = 'Train'
    else:
        hand = 'Val'
    table = PrettyTable(show_list)
    table.add_row(score_list)
    train_logger.log('{} Epoch: {}\n {}'.format(hand, epoch, table))
    print('{} Epoch: {}\n {}'.format(hand, epoch, table))

    if details:
        table = PrettyTable(['类别', 'F1', 'Precision', 'Recall'])
        for _ in range(number_class):
            table.add_row([str(_), f1_score_[_], precision_score_[_], recall_score_[_]])
        train_logger.log('{}\n'.format(table))
        print('{}\n'.format(table))

    return np.mean(f1_score_), accuracy_score_

# 训练模型过程
def train(model, train_data_loader, opt, ent_loss, epoch, train_logger, tqdm, apex, details, number_class):

    model.train()
    avg_loss = 0
    # 设置为空 因为numpy不能设置为空，所以创建完以后要用delete删除后，才为纯空  这里用来保存所有的预测标签 与 真实标签， 方便计算f1值
    pres_train = np.array([0], dtype=int)
    gths_train = np.array([0.00], dtype=float)
    pres_train = np.delete(pres_train, 0, axis=0)
    gths_train = np.delete(gths_train, 0, axis=0)
    count = 0
    # 每次遍历batch_size个数据   enumerate就是依次__getitem__的意思
    for _, (data, target) in enumerate(tqdm(train_data_loader)):
        # 1、避免梯度累加   每次都给它清零
        # 我们进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了。所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0
        opt.zero_grad()
        # 把data和label加入cuda中
        if torch.cuda.is_available():
            data, target = data.cuda(), target.long().cuda()
        else:
            data, target = data.cpu(), target.long().cpu()

        # 2、根据模型输出结果
        out = model(data)
        # 3、计算loss
        loss = ent_loss(out, target)
        # 4、根据loss进行反向传播
        # loss.backward()
        if apex:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # 5、optimizer.step用来更新参数
        opt.step()

        # 累加loss
        avg_loss = avg_loss + loss.item()
        count = count + 1

        softmax_pred = torch.nn.functional.softmax(out, 1)
        softmax_pred = torch.max(softmax_pred, 1)[1]

        pres_train = np.append(pres_train, softmax_pred.cpu().numpy())
        gths_train = np.append(gths_train, target.cpu().numpy())

    avg_loss = avg_loss / count
    f1_score_, accuracy_score_ = compute_weight_score(pres_train, gths_train, details, number_class, avg_loss, train_logger, epoch)

    return f1_score_, accuracy_score_, avg_loss


# 验证模型过程（其实是一个val的过程，因为这里存在真实标签）
def val(model, val_data_loader, epoch, train_logger, tqdm, details, number_class):

    print('Valing')
    model.eval()

    pres_val = np.array([0], dtype=int)
    gths_val = np.array([0.00], dtype=float)
    pres_val = np.delete(pres_val, 0, axis=0)
    gths_val = np.delete(gths_val, 0, axis=0)
    for _, (data, target) in enumerate(tqdm(val_data_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.long().cuda()
        else:
            data, target = data.cpu(), target.long().cpu()
        # 不再计算梯度
        with torch.no_grad():
            out = model(data)

        softmax_pred = torch.nn.functional.softmax(out, 1)
        softmax_pred = torch.max(softmax_pred, 1)[1]
        pres_val = np.append(pres_val, softmax_pred.cpu().numpy())
        gths_val = np.append(gths_val, target.cpu().numpy())

    f1_score_, accuracy_score_ = compute_weight_score(pres_val, gths_val, details, number_class, None, train_logger, epoch)

    return f1_score_, accuracy_score_, None