import os
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import OmegaConf

os.environ["TOKENIZERS_PARALLELISM"] = "false"
### offline wandb run to solve "conncetion error" problem
os.environ["WANDB_MODE"] = "online"

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
    import os

    # if 'hf_hub_cache_dir' in cfg:
        # TODO(jxma): we have to set this before loading hf transformers
        # import torch
        # torch.hub.set_dir(cfg.hf_hub_cache_dir)
        # os.environ['TRANSFORMERS_CACHE'] = cfg.hf_hub_cache_dir
        # os.environ['HUGGINGFACE_HUB_CACHE'] = cfg.hf_hub_cache_dir
        #TODO: temporary huggingface bug fix
        # os.environ['CURL_CA_BUNDLE'] = ''
    import common.io_utils as iu
    from common.misc import rgetattr
    from trainer.build import build_trainer

    naming_keys = [cfg.name]
    for name in cfg.naming_keywords:
        if name == "time":
            continue
        elif name == "task":
            naming_keys.append(cfg.task)
            datasets = rgetattr(cfg, f"data.{cfg.task.lower()}.dataset")
            dataset_names = "+".join([str(x) for x in datasets])
            naming_keys.append(dataset_names)
        elif name == "dataloader.batchsize":
            # naming_keys.append(f"b{rgetattr(cfg, name) * rgetattr(cfg, 'num_gpu')}")
            continue
        else:
            if str(rgetattr(cfg, name)) != "":
                naming_keys.append(str(rgetattr(cfg, name)))
    exp_name = "_".join(naming_keys)

    # Record the experiment
    cfg.exp_dir = Path(cfg.base_dir) / exp_name / (
            f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}" if "time" in cfg.naming_keywords else ""
    )
    if os.path.exists(cfg.exp_dir):
        if cfg.resume:
            print(f"Will resume from experiment {cfg.exp_dir}", flush=True)
        else:
            print(f'Will overwrite experiment {cfg.exp_dir}', flush=True)
    iu.make_dir(cfg.exp_dir)
    iu.save_yaml(OmegaConf.to_yaml(cfg), cfg.exp_dir / "config.yaml")
    print(f'Starting experiments: {exp_name}', flush=True)

    trainer = build_trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
