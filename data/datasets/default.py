from omegaconf import OmegaConf
from fvcore.common.registry import Registry
from torch.utils.data import Dataset, ConcatDataset


DATASET_REGISTRY = Registry("dataset")
DATASET_REGISTRY.__doc__ = """
Registry for datasets, which takes a list of dataset names and returns a dataset object.
Currently it performs similar as registering dataset loading functions, but remains in a
form of object class for future purposes.
"""

@DATASET_REGISTRY.register()
class DefaultDataset(Dataset):
    def __init__(self, cfg, split='train') -> None:
        super().__init__()
        # dataset parameters
        pass
        # combine dataset dicts
        self.data_dict = get_dataset_dicts(
                            cfg.data.pretrain.dataset,
                            cfg.task,
                            cfg,
                            split,
                            filter_empty=cfg.dataloader.filter_empty_annotations)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.

        Args:
            index (int): _description_
        """

        return self.data_dict[index]


def get_dataset_dicts(names, task, cfg, split='train', filter_empty=True):
    """_summary_

    Args:
        names (_type_): a list of str, each with DATASET-CAPSOURCE1-CAPSOURCE2-... 
        task (_type_): _description_
        cfg (_type_): _description_
        split (str, optional): _description_. Defaults to 'train'.
        filter_empty (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # dataset_names = ",".join(names)
    if isinstance(names, str):
        names = [names]

    # print('dataset names', names, task, f'{names[0]}{task}')

    assert len(names), names
    dataset_dicts = [DATASET_REGISTRY.get(f"{dataset_name.split('-')[0]}{task}")(cfg, split, dataset_name.split('-')[1:])
                                     for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), f"Dataset '{dataset_name}' is empty!"

    dataset_dicts = ConcatDataset(dataset_dicts)

    if filter_empty:
        pass

    assert dataset_dicts, f"No valid data found in {names}."
    print(f"Dataset loaded of length {len(dataset_dicts)}")
    return dataset_dicts

if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data.datasets.scannet import *
    cfg = OmegaConf.load('./configs/default.yaml')
    dataset_train = get_dataset_dicts(
                        cfg.data.pretrain.dataset,
                        cfg.task,
                        cfg,
                        'train',
    )
    print(len(dataset_train))
    print(dataset_train[0].keys())
    print(dataset_train[0]['sentence'])
    print(len(dataset_train[0]['obj_fts']))
    sys.exit()
