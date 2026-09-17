"""WarmupMultiStepLR ported near-verbatim from ice/utils/lr_scheduler.py. WarmupCosineLR is new,
added for examples/train_relational_prompts.py -- a self-contained, epoch-stepped linear-warmup + cosine-
decay schedule matching CLIP-ReID's actual Stage-1 schedule (timm's CosineLRScheduler, called via
`scheduler.step(epoch)`), without adding a timm dependency for one schedule shape.
"""
import math
from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of increasing integers. Got {}".format(milestones))
        if warmup_method not in ("constant", "linear"):
            raise ValueError('Only "constant" or "linear" warmup_method accepted, got {}'.format(warmup_method))

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup from warmup_lr_init to each param group's base_lr over warmup_epochs, then
    cosine decay from base_lr down to lr_min over the remaining epochs. Call .step() once per
    epoch (standard PyTorch convention), not per iteration."""

    def __init__(self, optimizer, max_epochs, warmup_epochs=5, warmup_lr_init=1e-5,
                 lr_min=1e-6, last_epoch=-1):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init
        self.lr_min = lr_min
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = float(epoch) / float(max(1, self.warmup_epochs))
            return [self.warmup_lr_init + alpha * (base_lr - self.warmup_lr_init)
                    for base_lr in self.base_lrs]
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
        progress = min(progress, 1.0)
        return [self.lr_min + 0.5 * (base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs]
