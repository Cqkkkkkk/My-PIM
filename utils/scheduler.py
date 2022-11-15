import math
import numpy as np

from config import cfg


class CosineDecayLRScheduler():
    def __init__(self, total_batches: int, decay_type: int = 1) -> None:

        total_batchs = cfg.optim.epochs * total_batches
        iters = np.arange(total_batchs - cfg.optim.warmup_batches)
        eps = 1e-12
        if decay_type == 1:
            base = np.array([(1 + math.cos(math.pi * t / total_batchs)) for t in iters])
            self.schedule = eps + 0.5 * cfg.optim.max_lr * base
        elif decay_type == 2:
            base = np.array([math.cos(7 * math.pi * t / (16 * total_batchs)) for t in iters])
            self.schedule = cfg.optim.max_lr * base
        else:
            raise ValueError("Undefined decay type")

        if cfg.optim.warmup_batches > 0:
            warmup_lr_schedule = np.linspace(1e-9, cfg.optim.max_lr, cfg.optim.warmup_batches)
            self.schedule = np.concatenate((warmup_lr_schedule, self.schedule))
    

    def step(self, iteration, optimizer):
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.schedule[iteration]


    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            if param_group["lr"] is not None:
                return param_group["lr"]




def eval_freq_schedule(epoch: int):
    if epoch >= cfg.optim.epochs * 0.95:
        cfg.train.eval_freq = 1
    elif epoch >= cfg.optim.epochs * 0.9:
        cfg.train.eval_freq = 1
    elif epoch >= cfg.optim.epochs * 0.8:
        cfg.train.eval_freq = 2


if __name__ == '__main__':
    pass
