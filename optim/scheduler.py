import math
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(step, warmup_step, total_step):
    if step <= warmup_step:
        return step / warmup_step
    return max(0.5 * (1 + math.cos((step - warmup_step) / (total_step - warmup_step) * math.pi)), 1e-5)


def warmup_exp(step, warmup_step, total_step, **kwargs):
    if step <= warmup_step:
        return step / warmup_step
    return kwargs["gamma"] ** (step * 1. / (total_step - warmup_step))


def warmup_cosine_instructblip(step, warmup_step, total_step):
    if step <= warmup_step:
        return 1e-3 + step / warmup_step * (1 - 1e-3)
    return 0.5 * (1 + math.cos((step - warmup_step) / (total_step - warmup_step) * math.pi))


def get_scheduler(cfg, optimizer, total_steps):
    lambda_func = lambda step: globals()[cfg.solver.sched.name](step, cfg.solver.sched.args.warmup_steps, total_steps)
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda_func)
