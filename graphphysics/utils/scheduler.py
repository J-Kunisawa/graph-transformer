from typing import List
import math

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine learning rate scheduler with a linear warmup.

    - Warmup: linearly increase LR from 0..base_lr over `warmup` iterations.
    - Cosine decay: after warmup, LR follows a cosine schedule down to `min_lr_factor * base_lr`.
    - Safe for edge cases: warmup=0, max_iters<=0, min_lr_factor bounds.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup: int,
        max_iters: int,
        min_lr_factor: float = 0.001,
        last_epoch: int = -1,
    ):
        if max_iters <= 0:
            raise ValueError("max_iters must be > 0")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if not (0.0 <= min_lr_factor <= 1.0):
            raise ValueError("min_lr_factor must be in [0.0, 1.0]")

        self.warmup = int(warmup)
        self.max_iters = int(max_iters)
        self.min_lr_factor = float(min_lr_factor)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Return the list of learning rates for each param group.
        _LRScheduler will set optimizer.param_groups[i]['lr'] to this value.
        """
        factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch: int) -> float:
        """
        Compute a scalar factor to multiply base_lr by.

        `epoch` is the scheduler's last_epoch (may be -1 initially). We convert it to a 1-based step count.
        """
        # convert last_epoch (-1,0,1,...) -> step (0,1,2,...)
        step = max(0, epoch + 1)

        # clamp step to [0, max_iters]
        step_clamped = min(step, self.max_iters)

        # linear warmup factor (0..1)
        if self.warmup > 0 and step_clamped <= self.warmup:
            warmup_frac = float(step_clamped) / float(self.warmup)
            # during pure warmup we still apply cosine baseline = 1.0 (so factor = warmup_frac)
            base_cos = 1.0
            lr_factor = warmup_frac * base_cos
        else:
            # cosine decay from step = warmup .. max_iters
            # map step_clamped in [0, max_iters] to cosine argument
            progress = float(step_clamped) / float(self.max_iters)
            cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_factor = cos_val

            # ensure warmup boundary continuity: when warmup==0, lr_factor already correct
            if self.warmup > 0 and step_clamped <= self.warmup:
                # already handled above, but keep for clarity
                pass

        # enforce minimum factor
        lr_factor = max(lr_factor, self.min_lr_factor)
        return lr_factor