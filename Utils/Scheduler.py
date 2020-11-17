from torch.optim import lr_scheduler
import torch

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                "got {}".format(warmup_method))

        self.target_lr = target_lr
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        N = self.max_iters - self.warmup_iters
        T = self.last_epoch - self.warmup_iters
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            else:
                raise ValueError("Unknown warmup type.")
            return [self.target_lr + (base_lr - self.target_lr) * warmup_factor for base_lr in self.base_lrs]
        factor = pow(1 - T / N, self.power)
        return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]

def get_lr_scheduler(type, opt, max_epoch, iter_epoch):
    if type == 'MultiStepLR':
        sch = lr_scheduler.MultiStepLR(opt, milestones=[int(max_epoch*0.25), int(max_epoch*0.5), int(max_epoch*0.75)], gamma=0.1)
    elif type == 'LambdaLR':
        lambda1 = lambda epoch: max_epoch // 30
        lambda2 = lambda epoch: 0.95 ** max_epoch
        sch = lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1, lambda2])
    elif type == 'StepLR':
        StepLR(optimizer, step_size=30, gamma=0.1)
    elif type == 'ExponentialLR':
        sch = lr_scheduler.ExponentialLR(opt, gamma=0.1)
    elif type == 'WarmupPolyLR':
        sch = WarmupPolyLR(optimizer=opt, max_iters=max_epoch*iter_epoch)

    else:
        sch = lr_scheduler.MultiStepLR(opt,
                                       milestones=[int(max_epoch * 0.25), int(max_epoch * 0.5), int(max_epoch * 0.75)],
                                       gamma=0.1)
    return sch