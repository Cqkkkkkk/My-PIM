import math
import numpy as np

from config import cfg


def cosine_decay(batchs: int, decay_type: int = 1):
    total_batchs = cfg.optim.epochs * batchs
    iters = np.arange(total_batchs - cfg.optim.warmup_batches)

    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (cfg.optim.max_lr - 1e-12) * (1 +
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = cfg.optim.max_lr * \
            np.array([math.cos(7*math.pi*t / (16*total_batchs))
                     for t in iters])
    else:
        raise ValueError("Not support this deccay type")

    if cfg.optim.warmup_batches > 0:
        warmup_lr_schedule = np.linspace(
            1e-9, cfg.optim.max_lr, cfg.optim.warmup_batches)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]


def adjust_lr(iteration, optimizer, schedule):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]
