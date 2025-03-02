from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate

from common.misc import rgetattr

from .datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY
from .datasets.default import DATASET_REGISTRY, get_dataset_dicts


def get_dataset(cfg, split, sources):

    dataset = get_dataset_dicts(sources,
                                cfg.task,
                                cfg,
                                split,
                                filter_empty=cfg.dataloader.filter_empty_annotations)
    # TODO: change wrapper name according to task
    if cfg.task in ['Pretrain']:
        dataset = DATASETWRAPPER_REGISTRY.get('MaskDatasetWrapper')(cfg, dataset)
    elif cfg.task in ['ScanRefer', 'Referit3D', 'ScanQA', 'MVReferit3D', 'SpatialRefer', 'ScanQAInstruction', 'SQA3DInstruction']:
        dataset = DATASETWRAPPER_REGISTRY.get('ScanFamilyDatasetWrapper')(cfg, dataset)
    elif cfg.task == "MVRecon":
        pass
    elif cfg.task in ['MVPretrain']:
        dataset = DATASETWRAPPER_REGISTRY.get('MaskMVDatasetWrapper')(cfg, dataset)
    else:
        raise NotImplementedError

    # Conduct voxelization
    if cfg.data.use_voxel:
        dataset = DATASETWRAPPER_REGISTRY.get('VoxelDatasetWrapper')(cfg, dataset)

    return dataset


def build_dataloader(cfg, split='train'):
    """_summary_
    Unittest:
        dataloader_train = build_dataloader(default_cfg, split='train')
        for _item in dataloader_train:
            print(_item.keys())

    Args:
        cfg (_type_): _description_
        split (str, optional): _description_. Defaults to 'train'.

    Returns:
        _type_: _description_
    """
    dataset_cfg = getattr(cfg.data, cfg.task.lower()).dataset
    sources = getattr(dataset_cfg, split)
    if split == 'train':
        dataset = get_dataset(cfg, split, sources)
        return DataLoader(dataset,
                          batch_size=cfg.dataloader.batchsize,
                          num_workers=cfg.dataloader.num_workers,
                          collate_fn=getattr(dataset, 'collate_fn', default_collate),
                          pin_memory=True, # TODO: Test speed
                          shuffle=True,
                          drop_last=True)
    else:
        loader_list = []
        for source in sources:
            dataset = get_dataset(cfg, split, source)
            loader_list.append(DataLoader(
                dataset,
                rgetattr(cfg, f"dataloader.batchsize_eval", cfg.dataloader.batchsize),
                num_workers=cfg.dataloader.num_workers,
                collate_fn=getattr(dataset, 'collate_fn', default_collate),
                pin_memory=True, # TODO: Test speed
                shuffle=False
            ))
        return loader_list


def get_dataset_leo(cfg, dataset_name, dataset_wrapper_name, dataset_wrapper_args, split):
    # just get dataset directly and then wrap it
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)

    if dataset_wrapper_name:
        if type(dataset_wrapper_name) != list:
            dataset_wrapper_name = [dataset_wrapper_name]
        if type(dataset_wrapper_args) != list:
            dataset_wrapper_args = [dataset_wrapper_args]

        for wrapper, wrapper_args in zip(dataset_wrapper_name, dataset_wrapper_args):
            dataset = DATASETWRAPPER_REGISTRY.get(wrapper)(cfg, dataset, wrapper_args)

    return dataset


def build_dataloader_leo(cfg, dataset_name, dataset_wrapper_name, dataset_wrapper_args, dataloader_args, split='train'):
    dataset = get_dataset_leo(cfg, dataset_name, dataset_wrapper_name, dataset_wrapper_args, split)
    return DataLoader(dataset,
                      batch_size=dataloader_args.batchsize,
                      num_workers=dataloader_args.num_workers,
                      collate_fn=getattr(dataset, 'collate_fn', default_collate),
                      pin_memory=True, # TODO: Test speed
                      shuffle=True if split == 'train' else False,
                      drop_last=True if split == 'train' else False) # TODO(jxma): maybe False for eval?


if __name__ == '__main__':
    pass
