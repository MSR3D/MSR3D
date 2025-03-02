import random

import numpy as np
import torch
from fvcore.common.registry import Registry
from torch.utils.data import Dataset, default_collate
from transformers import BertTokenizer

from ..data_utils import pad_tensors, random_point_cloud, random_word

# from modules.third_party.softgroup_ops.ops import functions as sg_ops

# import os
# os.environ['CURL_CA_BUNDLE'] = ''

DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")
DATASETWRAPPER_REGISTRY.__doc__ = """ """


@DATASETWRAPPER_REGISTRY.register()
class MaskDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        # tokenizer, max_seq_length=80, max_obj_len=80,
        #  mask_strategy='random', txt_mask_ratio=0.15, pc_mask_ratio=0.1
        assert getattr(cfg.data, cfg.task.lower()).args.mask_strategy in ['random']
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained(cfg.model.prompter.model.language.args.weights, do_lower_case=True)
        self.max_seq_length = getattr(cfg.data, cfg.task.lower()).args.max_seq_len
        self.max_obj_len = getattr(cfg.data, cfg.task.lower()).args.max_obj_len
        self.txt_mask_ratio = getattr(cfg.data, cfg.task.lower()).args.txt_mask_ratio
        self.pc_mask_ratio = getattr(cfg.data, cfg.task.lower()).args.pc_mask_ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # mask txt
        masked_txt_ids, masked_lm_labels = random_word(data_dict['txt_ids'], data_dict['txt_masks'],
                                                       self.tokenizer, self.txt_mask_ratio)
        data_dict['txt_ids'] = masked_txt_ids
        data_dict['masked_lm_labels'] = masked_lm_labels
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        data_dict['obj_fts'] = pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len,
                                                pad=1.0).float() # O, 1024, 6
        data_dict['obj_locs']= pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len,
                                                pad=0.0).float() # O, 3
        data_dict['obj_labels'] = pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                   pad=-100).long() # O
        # mask object, 0 means mask object, 1 means keep object
        obj_sem_masks = random_point_cloud(data_dict['obj_fts'], data_dict['obj_masks'],
                                           self.pc_mask_ratio)
        data_dict['obj_sem_masks'] = obj_sem_masks

        # # Scene pcds
        # data_dict["scene_pcds"] = torch.from_numpy(data_dict["scene_pcds"]).float()

        return data_dict

@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        # stokenizer, max_seq_length=80, max_obj_len=80
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained(cfg.model.prompter.model.language.args.weights, do_lower_case=True)
        self.max_seq_length = getattr(cfg.data, cfg.task.lower()).args.max_seq_len
        self.max_obj_len = getattr(cfg.data, cfg.task.lower()).args.max_obj_len

    def __len__(self):
        return len(self.dataset)

    def pad_tensors(self, tensors, lens=None, pad=0):
        assert tensors.shape[0] <= lens
        if tensors.shape[0] == lens:
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len,
                                                pad=1.0).float() # O, 1024, 6
        data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len,
                                                pad=0.0).float() # O, 3
        data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'], lens=self.max_obj_len,
                                                 pad=0.0).float() # O, 3
        data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                   pad=-100).long() # O
        # build sem mask, no mask
        data_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))
        # build label for refer
        data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
        if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
            data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(),
                                                          lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') is not None:
            data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'],
                                                                lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') is not None:
            data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'],
                                                                lens=self.max_obj_len, pad=0).long()
        # build label for qa
        if "answer_label" in data_dict:
            data_dict['answer_label'] = data_dict['answer_label'].long() # N, C
        return data_dict

@DATASETWRAPPER_REGISTRY.register()
class LeoScanFamilyDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset, dataset_wrapper_args):
        self.dataset = dataset
        self.max_obj_len = dataset_wrapper_args.get('max_obj_len', 60)

        self.msr3d_max_img_num = dataset_wrapper_args.get('msr3d_max_img_num', 10)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def pad_tensors(tensors, lens=None, pad=0):
        assert tensors.shape[0] <= lens
        if tensors.shape[0] == lens:
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]


        if 'obj_fts' in data_dict:
            data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len, pad=1.0).float()   # O, num_points, 6
            data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))   # O
            data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len, pad=0.0).float()   # O, 6
        if 'obj_labels' in data_dict:
            data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                    pad=-100).long() # O
        if 'tgt_object_label' in data_dict:
            data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        if 'tgt_object_id' in data_dict:
            data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
            if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
                data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(),
                                                            lens=self.max_obj_len, pad=0).long() # O
        if 'obj_boxes' in data_dict:
            data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'],
                                                     lens=self.max_obj_len,
                                                     pad=0.0).float()   # O, 6
            
        #### pad image list ####
        if 'msr3d_imgs' in data_dict:
            data_dict['msr3d_img_masks'] = (torch.arange(self.msr3d_max_img_num) < len(data_dict['msr3d_imgs']))
            if len(data_dict['msr3d_imgs']) == 0:
                data_dict['msr3d_imgs'] = torch.zeros((self.msr3d_max_img_num, 3, 224, 224))
            else:
                data_dict['msr3d_imgs'] = torch.stack(data_dict['msr3d_imgs'], dim=0)
                data_dict['msr3d_imgs'] = self.pad_tensors(data_dict['msr3d_imgs'], lens=self.msr3d_max_img_num, pad=0.0)

        return data_dict


    def collate_fn(self, batch):
        batch_dict = {}

        for key in batch[0].keys():
            values = [item[key] for item in batch]

            # Handle tensors (stack or pad them)
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() == 0:  # Scalar tensor
                    batch_dict[key] = torch.stack(values)
                else:
                    batch_dict[key] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0.0)

            # Handle lists of tensors (e.g., images)
            elif isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
                batch_dict[key] = [torch.stack(v) for v in values]

            # Handle string values
            elif isinstance(values[0], str):
                batch_dict[key] = values  # Keep as a list

            # Handle numerical lists (e.g., object indices)
            elif isinstance(values[0], list) and isinstance(values[0][0], (int, float)):
                max_len = max(len(v) for v in values)
                padded_values = [v + [0] * (max_len - len(v)) for v in values]
                batch_dict[key] = torch.tensor(padded_values)

            else:
                batch_dict[key] = values  # Keep as is for unsupported types

        return batch_dict
