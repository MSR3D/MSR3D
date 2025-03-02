import copy as cp
from datetime import datetime, timedelta
from pathlib import Path

# from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate import DistributedDataParallelKwargs
from common.misc import CustomAccelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from fvcore.common.registry import Registry
import torch
import wandb

from common.io_utils import make_dir
from data.build import build_dataloader
from evaluator.build import build_eval
from model.build import build_model
from optim.build import build_optim


TRAINER_REGISTRY = Registry("Trainer")


class Tracker():
    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1
        self.loader_step = 0

    def step_loader(self):
        self.loader_step += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.note}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        self.run_id = wandb.util.generate_id()
        self.epoch = 0
        self.loader_step = 0

    def state_dict(self):
        return {"run_id": self.run_id, "epoch": self.epoch, "exp_name": self.exp_name, "loader_step": self.loader_step}

    def load_state_dict(self, state_dict):
        state_dict = cp.deepcopy(state_dict)
        self.run_id = state_dict["run_id"]
        self.epoch = state_dict["epoch"]
        self.loader_step = state_dict["loader_step"]
        self.exp_name = state_dict["exp_name"]


class BaseTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.debug = cfg.debug.flag

        # Initialize accelerator
        self.exp_tracker = Tracker(cfg)
        # There is bug in logger setting, needs fixing from accelerate side
        self.logger = get_logger(__name__)
        self.mode = cfg.mode
        self.need_resume = cfg.resume
        self.save_freq = cfg.save_frequency

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]

        self.accelerator = CustomAccelerator(
            gradient_accumulation_steps=cfg.solver.get("gradient_accumulation_steps", 1),
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )

        keys = ["train", "val"] if cfg.task == "Pretrain" else ["train", "val", "test"]
        self.data_loaders = {key : build_dataloader(cfg, split=key) for key in keys}
        self.model = build_model(cfg)
        self.optimizer, self.scheduler = build_optim(cfg, self.model.get_opt_params(),
                                                     total_steps=len(self.data_loaders["train"]) * cfg.solver.epochs)
        self.evaluator = build_eval(cfg, self.accelerator)
        self.inference_mode = cfg.model.llm.inference_mode

        self.learn_params_list = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.learn_params_list.append(n)

        assert len(self.learn_params_list) == len(self.model.get_opt_params())

        # Training details
        self.epochs = cfg.solver.epochs
        self.eval_interval = cfg.solver.eval_interval
        self.num_batch_eval = cfg.dataloader.num_batch_eval
        self.total_steps = len(self.data_loaders["train"]) * cfg.solver.epochs
        self.grad_norm = cfg.solver.get("grad_norm")

        # Load pretrain model weights
        self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
        if self.pretrain_ckpt_path.exists() and cfg.pretrain_ckpt_path != "":
            self.load_pretrain()

        # Accelerator preparation
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        for name, loader in self.data_loaders.items():
            self.data_loaders[name] = self.accelerator.prepare(loader)
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # Check if resuming from previous checkpoint is needed
        self.ckpt_path = Path(cfg.ckpt_path) if cfg.ckpt_path != "" else Path(cfg.exp_dir) / "ckpt" / "best.pth"
        if self.need_resume:
            self.resume()

        # Misc
        if not self.debug:
            self.accelerator.init_trackers(
                    project_name=cfg.name,
                    config={"num_epochs": cfg.solver.epochs, "batch_size": cfg.dataloader.batchsize, "lr": cfg.solver.lr},
                    init_kwargs={
                        "wandb": {
                            "name": self.exp_tracker.exp_name, "entity": cfg.logger.entity,
                            "id": self.exp_tracker.run_id, "resume": True
                        }
                    }
                )

    def forward(self, data_dict, inference=False):
        is_generation_mode = (self.inference_mode == 'generation')
        is_model_distributed = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        if inference:
            if is_generation_mode:
                if is_model_distributed:
                    return self.model.module.generate(data_dict)
                else:
                    return self.model.generate(data_dict)
            else:
                if is_model_distributed:
                    return self.model.module.predict_answers(
                        data_dict, answer_list=list(self.evaluator.answer_vocab.vocab.keys())
                    )
                else:
                    return self.model.predict_answers(
                        data_dict, answer_list=list(self.evaluator.answer_vocab.vocab.keys())
                    )
        else:
            return self.model(data_dict)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        raise NotImplementedError

    def eval_step(self):
        raise NotImplementedError

    def test_step(self):
        raise NotImplementedError

    def log(self, results, step=0, mode="train"):
        if not self.debug:
            log_dict = {}
            for key, val in results.items():
                if 'loss' in key or 'llm' in key or key == 'target_metric':
                    log_dict[f"{mode}/{key}"] = val
            if mode == "train":
                lrs = self.scheduler.get_lr()
                for i, lr in enumerate(lrs):
                    log_dict[f"lr/group_{i}"] = lr
            self.accelerator.log(log_dict)

    def save(self, name):
        make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))

    def resume(self):
        if self.ckpt_path.exists():
            print(f"Resuming from {str(self.ckpt_path)}")
            # self.logger.info(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            # self.logger.info(f"Successfully resumed from {self.ckpt_path}")
            print(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")

    def load_pretrain(self):
        self.accelerator.print(f"Loading pretrained weights from {str(self.pretrain_ckpt_path)}")
        model_weight_path = self.pretrain_ckpt_path / "pytorch_model.bin"
        visual_prompter_sd = torch.load(str(model_weight_path), map_location="cpu")
        for k in ['qa_head.answer_cls.3.weight', 'qa_head.answer_cls.3.bias']:
            # remove unrelated keys
            del visual_prompter_sd[k]
        self.model.visual_prompter.load_state_dict(visual_prompter_sd, strict=False)
        self.accelerator.print(f"Successfully loaded from {str(self.pretrain_ckpt_path)}")

    def save_func(self, path):
        self.accelerator.save_state(path)

    def run(self):
        raise NotImplementedError


def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
