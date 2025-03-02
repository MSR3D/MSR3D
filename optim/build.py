from common.type_utils import cfg2dict

from optim.loss.loss import Loss
from optim.scheduler import get_scheduler


def build_optim(cfg, params, total_steps):
    # loss = Loss(cfg)
    print(f"Use {cfg.solver.optim.name} optimizer")
    if cfg.solver.optim.name == 'Lamb':
        import torch_optimizer as optim
        optimizer = optim.Lamb(params, **cfg2dict(cfg.solver.optim.args))
    else:
        import torch.optim as optim
        optimizer = getattr(optim, cfg.solver.optim.name)(params, **cfg2dict(cfg.solver.optim.args))
    scheduler = get_scheduler(cfg, optimizer, total_steps)
    return optimizer, scheduler
