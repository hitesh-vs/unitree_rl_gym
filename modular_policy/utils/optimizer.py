import numpy as np
from modular_policy.config import cfg


def lr_fun_cos(cur_iter):
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_iter / cfg.PPO.MAX_ITERS))
    return (1.0 - cfg.PPO.MIN_LR) * lr + cfg.PPO.MIN_LR


def lr_fun_lin(cur_iter):
    lr = 1.0 - cur_iter / cfg.PPO.MAX_ITERS
    return (1.0 - cfg.PPO.MIN_LR) * lr + cfg.PPO.MIN_LR


def lr_fun_constant(cur_iter):
    return 1.0


def get_lr_fun():
    lr_fun = "lr_fun_" + cfg.PPO.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.PPO.LR_POLICY
    return globals()[lr_fun]


def get_iter_lr(cur_iter):
    lr = get_lr_fun()(cur_iter) * cfg.PPO.BASE_LR
    if cur_iter < cfg.PPO.WARMUP_ITERS:
        alpha = cur_iter / cfg.PPO.WARMUP_ITERS
        warmup_factor = cfg.PPO.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    print(f"lr={lr} for iter {cur_iter}")
    return lr


def set_lr(optimizer, new_lr, scale):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = new_lr * scale[i]