import logging
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from accelerate import DistributedDataParallelKwargs
from common.misc import CustomAccelerator
from accelerate.logging import get_logger
from accelerate.utils import (DeepSpeedPlugin, InitProcessGroupKwargs,
                              ProjectConfiguration, set_seed)
from tqdm import tqdm

from common.io_utils import make_dir
from data.build import build_dataloader_leo
from data.data_utils import vis_scene_qa
from evaluator.build import build_eval_leo
from model.build import build_model
from optim.build import build_optim
from trainer.build import TRAINER_REGISTRY, BaseTrainer, Tracker
from IPython import embed
from data.data_utils import path_verify, save_to_json

logger = logging.getLogger(__name__)

def latest_checkpoint(path):
    if not os.path.exists(path):
        return ""
    checkpoints = [os.path.join(path, f) for f in os.listdir(path) if "checkpoint" in f]
    if len(checkpoints) == 0:
        return ""
    return max(checkpoints, key=os.path.getctime)

@TRAINER_REGISTRY.register()
class LeoTrainer(BaseTrainer):
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.debug = cfg.debug.flag
        self.debug_test = cfg.debug.get('debug_test', False)
        self.exp_dir = cfg.exp_dir
        self.cfg_raw = cfg

        # Initialize accelerator
        self.exp_tracker = Tracker(cfg)
        # There is bug in logger setting, needs fixing from accelerate side
        self.logger = get_logger(__name__)
        self.mode = cfg.mode

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]

        # accelerator checkpointing
        self.need_resume = cfg.resume
        self.save_freq = cfg.save_frequency
        # TODO(jxma): accelerator will save checkpoints to `checkpoints` folder
        self.ckpt_path = cfg.ckpt_path if cfg.ckpt_path != "" else latest_checkpoint(os.path.join(self.exp_dir, "checkpoints"))

        self.accelerator = CustomAccelerator(
            project_config=ProjectConfiguration(
                project_dir=self.exp_dir,
                automatic_checkpoint_naming=True,
                total_limit=1, # TODO(jxma): hard-coded for now, save up to 1 recent checkpoints as it will save everything (inc. LLM...)
                ),
            gradient_accumulation_steps=cfg.solver.get("gradient_accumulation_steps", 1),
            # deepspeed_plugin=DeepSpeedPlugin(
            #     gradient_accumulation_steps=cfg.solver.get("gradient_accumulation_steps", 1),
            #     zero_stage=cfg.solver.get("zero_stage", 0),
            #     offload_optimizer_device=cfg.solver.get("offload_optimizer_device", None),
            #     offload_param_device=cfg.solver.get("offload_param_device", None),
            # ),
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs
        )

        # dataset, data loader, and evaluator
        self.eai_task_sources = ['hm3d', 'mp3d', 'cliport']
        self.data_loaders = {'train': {}, 'val': {}, 'test': {}}
        self.evaluators = {}
        self.eval_metrics = {}
        for task_name in cfg.task.keys():
            for mode in cfg.task[task_name].mode:
                # some tasks are evaluator only, ex. EAI
                if 'dataset' not in cfg.task[task_name]:
                    break
                self.data_loaders[mode][task_name] = build_dataloader_leo(
                    cfg,
                    cfg.task[task_name].dataset,
                    cfg.task[task_name].dataset_wrapper,
                    cfg.task[task_name].dataset_wrapper_args,
                    cfg.task[task_name].train_dataloader_args if mode == "train" else cfg.task[task_name].eval_dataloader_args,
                    mode)
            if 'evaluator' in cfg.task[task_name]:
                self.evaluators[task_name] = build_eval_leo(cfg, cfg.task[task_name].evaluator, task_name)
                self.eval_metrics[task_name] = 0

        self.overall_best_result = 0

        # FIXME(jxma): assume only 1 training dataset
        assert len(self.data_loaders['train']) == 1

        self.model = build_model(cfg)
        # FIXME(jxma): hard-coded
        self.total_steps = len(list(self.data_loaders["train"].values())[0]) * cfg.solver.epochs
        self.optimizer, self.scheduler = build_optim(cfg, self.model.get_opt_params(),
                                                     total_steps=self.total_steps)
        self.inference_mode = cfg.model.llm.inference_mode

        self.learn_params_list = []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.learn_params_list.append(n)

        assert len(self.learn_params_list) == len(self.model.get_opt_params())

        # Training details
        self.epochs = cfg.solver.epochs
        self.eval_interval = cfg.solver.eval_interval
        self.num_batch_eval = cfg.solver.num_batch_eval
        self.grad_norm = cfg.solver.get("grad_norm")

        # Load pretrain model weights
        self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
        if self.pretrain_ckpt_path.exists() and cfg.pretrain_ckpt_path != "":
            self.load_pretrain()

        # Accelerator preparation
        # TODO(jxma): this is weird, but DeepSpeed requires passing all dataloader in prepare()
        all_loaders, all_loader_keys = [], []
        for mode, loaders in self.data_loaders.items():
            for task, loader in loaders.items():
                all_loader_keys.append((mode, task))
                all_loaders.append(loader)
        ret = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, *all_loaders)
        self.model, self.optimizer, self.scheduler = ret[0], ret[1], ret[2]
        for k, v in zip(all_loader_keys, ret[3:]):
            self.data_loaders[k[0]][k[1]] = v
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        # resume, if needed
        if self.mode == "train":
            if self.need_resume and self.ckpt_path:
                self.accelerator.load_state(self.ckpt_path)
                print(f'Resuming from {self.ckpt_path}...', flush=True)
                print(f'Restored state, exp name: {self.exp_tracker.exp_name}, run id: {self.exp_tracker.run_id}, epoch: {self.exp_tracker.epoch}, loader step: {self.exp_tracker.loader_step}', flush=True)
            else:
                print(f'Starting from scratch...', flush=True)

        # Misc (must put after resume)
        if not self.debug:
            self.accelerator.init_trackers(
                    project_name=cfg.name,
                    # FIXME(jxma): hard coded
                    config=dict(cfg),
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
                assert False, "Reterival mode is not supported."
        else:
            return self.model(data_dict)

    def recurrent_forward_action(self, data_dict):
        # TODO(jxma): make sure this function is called recurrently and
        # 'past_action' needs to be relayed to the next call, unless a
        # reset is triggered.
        assert False, "Not implemented yet, but just put past action into text condition"
        is_model_distributed = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        if is_model_distributed:
            return self.model.module.generate(data_dict)
        else:
            return self.model.generate(data_dict)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        self.model.train()

        loader = list(self.data_loaders["train"].values())[0]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))

        # TODO(jxma): we may have to skip the batch that leads to NaN as well
        if self.exp_tracker.loader_step > 0:
            print(f'Skip the first {self.exp_tracker.loader_step} batches.', flush=True)
            loader = self.accelerator.skip_first_batches(loader, self.exp_tracker.loader_step)
            pbar.update(self.exp_tracker.loader_step)

        # import ipdb; ipdb.set_trace()
        for data_dict in loader:
            with self.accelerator.accumulate(self.model):
                # import ipdb; ipdb.set_trace()
                step = epoch * len(loader) + self.exp_tracker.loader_step
                data_dict['cur_step'] = step
                data_dict['total_steps'] = self.total_steps
                is_txt_data = [(s not in self.eai_task_sources) for s in data_dict['source']]
                is_eai_data = [(s in self.eai_task_sources) for s in data_dict['source']]
                # forward
                data_dict = self.forward(data_dict, inference=False)
                # calculate loss
                loss = data_dict['loss']
                loss_all = loss.mean()
                loss_txt = loss[is_txt_data]
                loss_eai = loss[is_eai_data]
                # optimize
                # save the inputs_embeds and attention_mask for visualization
                if self.cfg_raw.debug.flag and self.cfg_raw.debug.get("save_tensor_flag", False):
                    if data_dict['cur_step'] % 5000 == 0 or data_dict['cur_step'] < 20:
                        save_path = os.path.join(self.cfg_raw.base_dir, '_'.join([self.cfg_raw.name, self.cfg_raw.note]), "debug_tensor", f"cur_step_{data_dict['cur_step']}")
                        path_verify(save_path)
                        loss_all_path = os.path.join(save_path, "loss_all.pt")
                        torch.save(loss_all, loss_all_path)
                    
                self.backward(loss_all)
                # record
                loss_dict = {'loss': loss_all}
                if len(loss_txt) > 0:
                    loss_dict.update({'loss_txt': loss_txt.mean()})
                if len(loss_eai) > 0:
                    loss_dict.update({'loss_eai': loss_eai.mean()})
                self.log(loss_dict, step, mode='train')
                self.exp_tracker.step_loader()
                # checkpointing
                # if step % self.save_freq == 0 and self.accelerator.is_main_process:
                #     self.save(f"epoch{epoch}_step{step}.pth")
                #     self.accelerator.save_state()
                pbar.update(1)

    # def batch_detokenize(self, data_dict):
    #     if 'output_tokens' not in data_dict:
    #         return data_dict
        
    #     import random
    #     is_model_distributed = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
    #     max_id = self.model.module.llm_tokenizer.vocab_size
    #     # data_dict['output_tokens'][data_dict['output_tokens'] >= max_id] = 2
    #     # data_dict['output_tokens'][data_dict['output_tokens'] < 0] = 2
    #     if torch.any(data_dict['output_tokens'] >= max_id) or torch.any(data_dict['output_tokens'] < 0):
    #         print(f'Error in decoding token IDS {data_dict["output_tokens"]}')
    #         data_dict['output_tokens'][2] = 2
    #         save_path = os.path.join(self.cfg_raw.base_dir, '_'.join([self.cfg_raw.name, self.cfg_raw.note]), "debug_tensor_error", str(random.random()))
    #         path_verify(save_path)
    #         torch.save(data_dict, os.path.join(save_path, f"error_data_dict.pt"))
    #         output_text = ['']*len(data_dict['output_tokens'])
    #         data_dict['output_text'] = output_text
    #     else:
    #         if is_model_distributed:
    #             if torch.any(data_dict['output_tokens'] < 0):
    #                 print(f'Error in decoding token IDS {data_dict["output_tokens"]}')
    #                 output_text = ['']*len(data_dict['output_tokens'])
    #                 save_path = os.path.join(self.cfg_raw.base_dir, '_'.join([self.cfg_raw.name, self.cfg_raw.note]), "debug_tensor_error", str(random.random()))
    #                 path_verify(save_path)
    #                 torch.save(data_dict, os.path.join(save_path, f"error_data_dict.pt"))
    #             # HACK
    #             try:
    #                 output_text = self.model.module.llm_tokenizer.batch_decode(data_dict['output_tokens'], skip_special_tokens=True)
    #             except:
    #                 print(f'Error in decoding token IDS {data_dict["output_tokens"]}')
    #                 output_text = ['']*len(data_dict['output_tokens'])
    #                 save_path = os.path.join(self.cfg_raw.base_dir, '_'.join([self.cfg_raw.name, self.cfg_raw.note]), "debug_tensor_error", str(random.random()))
    #                 path_verify(save_path)
    #                 torch.save(data_dict, os.path.join(save_path, f"error_data_dict.pt"))
    #         else:
    #             if torch.any(data_dict['output_tokens'] < 0):
    #                 print(f'Error in decoding token IDS {data_dict["output_tokens"]}')
    #                 output_text = ['']*len(data_dict['output_tokens'])
    #                 save_path = os.path.join(self.cfg_raw.base_dir, '_'.join([self.cfg_raw.name, self.cfg_raw.note]), "debug_tensor_error", str(random.random()))
    #                 path_verify(save_path)
    #                 torch.save(data_dict, os.path.join(save_path, f"error_data_dict.pt"))
    #             try:  
    #                 output_text = self.model.llm_tokenizer.batch_decode(data_dict['output_tokens'], skip_special_tokens=True)
    #             except:
    #                 print(f'Error in decoding token IDS {data_dict["output_tokens"]}')
    #                 output_text = ['']*len(data_dict['output_tokens'])
    #                 save_path = os.path.join(self.cfg_raw.base_dir, '_'.join([self.cfg_raw.name, self.cfg_raw.note]), "debug_tensor_error", str(random.random()))
    #                 path_verify(save_path)
    #                 torch.save(data_dict, os.path.join(save_path, f"error_data_dict.pt"))
    #         output_text = [text.strip() for text in output_text]
    #         data_dict['output_text'] = output_text

    #         # remove redundant `eos_token`
    #         slim_tokens = []
    #         for i in range(data_dict['output_tokens'].shape[0]):
    #             tokens = data_dict['output_tokens'][i].detach().cpu().numpy()
    #             if 2 in tokens:
    #                 first_eos_idx = (tokens == 2).nonzero()[0][0]
    #                 slim_tokens.append(tokens[:first_eos_idx+1])
    #             else:
    #                 slim_tokens.append(tokens)
    #         data_dict['output_tokens'] = slim_tokens
    #     return data_dict

    def batch_detokenize(self, data_dict):
        if 'output_tokens' not in data_dict:
            return data_dict

        is_model_distributed = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        if is_model_distributed:
            output_text = self.model.module.llm_tokenizer.batch_decode(data_dict['output_tokens'], skip_special_tokens=True)
        else:
            output_text = self.model.llm_tokenizer.batch_decode(data_dict['output_tokens'], skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        # HACK
        # print("**********batch_output_text**********")
        # print(output_text)
        data_dict['output_text'] = output_text

        # remove redundant `eos_token`
        slim_tokens = []
        for i in range(data_dict['output_tokens'].shape[0]):
            tokens = data_dict['output_tokens'][i].detach().cpu().numpy()
            if 2 in tokens:
                first_eos_idx = (tokens == 2).nonzero()[0][0]
                slim_tokens.append(tokens[:first_eos_idx+1])
            else:
                slim_tokens.append(tokens)
        data_dict['output_tokens'] = slim_tokens
        return data_dict

    @torch.no_grad()
    def val_step(self, epoch):
        self.model.eval()
        # scan through each loader and evaluator
        for task_name in self.evaluators.keys():
            # dataset based evaluation
            if task_name in self.data_loaders['val']:
                loader = self.data_loaders["val"][task_name]
                pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
                for i, data_dict in enumerate(loader):
                    if i >= self.num_batch_eval:
                        break

                    data_dict = self.forward(data_dict, inference=True)

                    data_dict_non_tensor = {k : v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
                    data_dict_non_tensor = CustomAccelerator.gather_for_metrics(self.accelerator, data_dict_non_tensor)
                    data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
                    data_dict = CustomAccelerator.gather_for_metrics(self.accelerator, data_dict)
                    data_dict.update(data_dict_non_tensor)

                    data_dict = self.batch_detokenize(data_dict)

                    self.evaluators[task_name].update(data_dict)
                    pbar.update(1)
                _, results = self.evaluators[task_name].record()
            # TODO(jxma): other evaluations (without dataloader), assumed to be EAI
            elif task_name in self.data_loaders['test']:
                continue
            else:
                _, results = self.evaluators[task_name].record_eai(self.recurrent_forward_action, split='val')

            self.eval_metrics[task_name] = results['target_metric']
            self.log(results, epoch, mode="val", task=task_name)
            print(results, flush=True)
            self.evaluators[task_name].reset()

        # FIXME(jxma): we should have better strategies
        if sum(list(self.eval_metrics.values())) > self.overall_best_result:
            is_best = True
            self.overall_best_result = sum(list(self.eval_metrics.values()))
        else:
            is_best = False
        return is_best

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        # scan through each loader and evaluator
        all_results = {}
        # import ipdb; ipdb.set_trace()
        for task_name in self.evaluators.keys():
            # dataset based evaluation
            if task_name in self.data_loaders['test']:
                loader = self.data_loaders["test"][task_name]
                pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
                for i, data_dict in enumerate(loader):
                    data_dict = self.forward(data_dict, inference=True)

                    data_dict_non_tensor = {k : v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
                    data_dict_non_tensor = CustomAccelerator.gather_for_metrics(self.accelerator, data_dict_non_tensor)
                    data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
                    data_dict = CustomAccelerator.gather_for_metrics(self.accelerator, data_dict)
                    data_dict.update(data_dict_non_tensor)

                    data_dict = self.batch_detokenize(data_dict)

                    self.evaluators[task_name].update(data_dict)
                    pbar.update(1)
                _, results = self.evaluators[task_name].record(split='test')
            # other evaluations (without dataloader)
            else:
                _, results = self.evaluators[task_name].record_eai(self.recurrent_forward_action, split='test')

            self.log(results, 0, mode="test", task=task_name)
            print(results, flush=True)
            self.evaluators[task_name].reset()
            all_results[task_name] = results
        return all_results

    @torch.no_grad()
    def vis_step(self):
        pass

    def log(self, results, step=0, mode="train", task='default', also_log_lr=True):
        if not self.debug:
            log_dict = {}
            for key, val in results.items():
                log_dict[f"{mode}/{task}/{key}"] = val

            if mode == "train" and also_log_lr:
                lrs = self.scheduler.get_lr()
                for i, lr in enumerate(lrs):
                    log_dict[f"lr/group_{i}"] = lr

            self.accelerator.log(log_dict)

    def save(self, name):
        self.save_func(str(os.path.join(self.exp_dir, name)))

    def load_pretrain(self):
        self.accelerator.print(f"Loading pretrained weights from {str(self.pretrain_ckpt_path)}")
        self.load_model(self.pretrain_ckpt_path)
        self.accelerator.print(f"Successfully loaded from {str(self.pretrain_ckpt_path)}")

    def save_func(self, path):
        make_dir(path)
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) \
                           else self.model.state_dict()
        keys_list = list(model_state_dict.keys())
        for k in keys_list:
            if k not in self.learn_params_list:
                del model_state_dict[k]

        torch.save(model_state_dict, os.path.join(path, 'pytorch_model.bin'))

    def load_model(self, path):
        model_state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
        is_model_distributed = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        if is_model_distributed:
            self.model.module.load_state_dict(model_state_dict, strict=False)
        else:
            self.model.load_state_dict(model_state_dict, strict=False)

    def run(self):
        if self.mode == "train":
            # if self.accelerator.is_main_process:
            #     self.accelerator.save_state()
            start_epoch = self.exp_tracker.epoch
            # self.val_step(epoch=start_epoch)   # this is quite slow
            for epoch in range(start_epoch, self.epochs):
                if not self.debug_test:
                    self.train_step(epoch)
                    # must step after train_step so restoring loader step won't work
                    self.exp_tracker.step()

                    if self.accelerator.is_main_process:
                        self.save(f"epoch{epoch}.pth")

                # if self.accelerator.is_main_process:
                if (epoch + 1) % self.eval_interval == 0:
                    is_best = self.val_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}] finished eval, is_best: {is_best}")

                    self.accelerator.wait_for_everyone()
                    if is_best and self.accelerator.is_main_process:
                        self.save("best.pth")

            # load best checkpoint for test
            self.accelerator.print(f"Load best ckpt from {str(os.path.join(self.exp_dir, 'best.pth'))} for testing")
            self.load_model(os.path.join(self.exp_dir, 'best.pth'))
            self.accelerator.print(f"Successfully loaded from {str(os.path.join(self.exp_dir, 'best.pth'))}")
        else:
            if os.path.exists(os.path.join(self.exp_dir, 'best.pth')):
                self.accelerator.print(f"Load best ckpt from {str(os.path.join(self.exp_dir, 'best.pth'))} for testing")
                self.load_model(os.path.join(self.exp_dir, 'best.pth'))
                self.accelerator.print(f"Successfully loaded from {str(os.path.join(self.exp_dir, 'best.pth'))}")
            else:
                # directly load a specified checkpoint for test
                print(f'Start testing with checkpoint: {self.pretrain_ckpt_path}')

        self.test_step()
        # self.vis_step()

        self.accelerator.end_training()
