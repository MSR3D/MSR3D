import os
import collections
import json
import random
from copy import deepcopy
import jsonlines
from tqdm import tqdm
import numpy as np
import torch

from .default import DATASET_REGISTRY
from ..data_utils import build_rotate_mat
from data.data_utils import (VICUNA_ACTION_TOKENS)
from .scannet_base import ScanNetBase
from scipy.spatial.transform import Rotation as R

ONESTEPNAVI_ACTION_SPACE = {
    'move_forward': 0,
    'turn_left': 1,
    'move_backward': 2,
    'turn_right': 3,
    'turn_left_forward': 4,
    'turn_left_backward': 5,
    'turn_right_backward': 6,
    'turn_right_forward': 7,
}

ONESTEPNAVI_ACTION_SPACE_TOKENIZE = {
    k: v for k, v in zip(list(ONESTEPNAVI_ACTION_SPACE.values()), list(VICUNA_ACTION_TOKENS.keys())[:len(ONESTEPNAVI_ACTION_SPACE)])
}

NAVI_ACTION_POOL = [
    "What action should I take next step?",
]

# one_data_for_training = {
#         "location" : location,
#         "orientation" : quaternion.tolist(),
#         "situation_multimodal" : situation_desp,
#         "situation_text" : situation_text,
#         "interaction" : interaction,
#         "instruction" : instruction,
#         "action" : {
            #     "four_direction": [
            #         1,
            #         "turn left"
            #     ],
            #     "eight_direction": [
            #         1,
            #         "turn to foward left"
            #     ],
            #     "angle": 60.37389999185825
            # },
#         "meta_data" : one_data
#     }
# mapping = {
#     0 : "move forward", 1 : "turn left", 2 : "move backward", 3 : "turn right", 4 : "move forward" 
# }
# mapping = {
#     0 : "move forward", 1 : "turn to foward left", 2 : "turn left", 3 : "turn to back left", 4 : "move backward", 
#     5 : "turn to back right", 6 : "turn right", 7 : "turn to foward right", 8 : "move forward"
# } 

@DATASET_REGISTRY.register()
class ScanNetOneStepNavi(ScanNetBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.dataset_cfg = cfg.data.next_step_navigation.args
        
        self.num_points = self.dataset_cfg.get('num_points', 1024)
        self.max_obj_len = self.dataset_cfg.get('max_obj_len', 60)
        self.pc_type = self.dataset_cfg.get('pc_type', 'gt')
        self.action_type = self.dataset_cfg.get('action_type', 'four_direction')
        self.modality_type = self.dataset_cfg.get('modality_type', 'multimodal')

        assert self.pc_type in ['gt', 'pred']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        if self.split == 'test':
            self.split = 'val'

        self.action_mapping = {
            "four_direction": {0:0, 1:1, 2:2, 3:3, 4:0},
            "eight_direction": {0:0, 2:1, 4:2, 6:3, 8:0, 1:4, 3:5, 5:6, 7:7},
        }

        self.scan_ids = self._load_split(cfg, self.split)
        anno_file_path = os.path.join(cfg.data.msnn_base, 'msnn_scannet.json')
        with open(anno_file_path, 'r') as f:
            anno_info_all = json.load(f)

        print(f"Loading ScanNet ScanNetOneStepNavi {split}-set language")
        self.data, self.scan_ids = self._load_lang(anno_info_all, self.scan_ids)
        if cfg.debug.flag:
            self.data = self.data[:cfg.debug.debug_size]
        print(f"Finish loading ScanNetOneStepNavi {split}-set language")
        
        # load scans
        print(f"Loading ScanNet ScanNetOneStepNavi {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type, self.pc_type == 'gt')
        print(f"Finish loading ScanNet ScanNetOneStepNavi {split}-set data")
    
    def _load_lang(self, anno_info_all, select_scan_ids):
        output_list = []
        scan_ids = []
        for scan_id, samples_one_scene in anno_info_all.items():
            if scan_id not in select_scan_ids:
                continue
            scan_ids.append(scan_id)
            for one_sample in samples_one_scene.values():
                one_sample['insts'] = [int(x) for x in one_sample['insts']]
                output_list.append(one_sample)
        scan_ids = list(set(scan_ids))
        return output_list, scan_ids
    
    def __len__(self):
        return len(self.data)

    # get inputs for scene encoder
    def preprocess_pcd(self, obj_pcds, return_anchor = False, rot_aug = True, situation = None):
        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug=rot_aug)

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        for i, obj_pcd in enumerate(obj_pcds):
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())

            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))
            if return_anchor and i == 0:
                # Select a loc within the obj bbox as the anchor.
                anchor_loc = obj_pcd[:, :3].min(0) + np.random.rand(3) * obj_size

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)
            obj_pcd = obj_pcd[pcd_idxs]

            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.sqrt((obj_pcd[:, :3]**2).sum(1)).max()
            if max_dist < 1e-6:   # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        if return_anchor:
            anchor_loc = torch.from_numpy(anchor_loc)
        else:
            anchor_loc = torch.zeros(3).float()

        output_dict = {
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            'anchor_loc': anchor_loc,
        }
        
        if situation is not None:
            if rot_matrix is None:
                output_dict["situation"] = situation
            else:
                pos, ori = situation
                pos = np.array(pos)
                ori = np.array(ori)
                pos_new = pos.reshape(1, 3) @ rot_matrix.transpose()
                pos_new = pos_new.reshape(-1)
                ori_new = R.from_quat(ori).as_matrix()
                ori_new = rot_matrix @ ori_new
                ori_new = R.from_matrix(ori_new).as_quat()
                ori_new = ori_new.reshape(-1)
                output_dict["situation"] = (pos_new, ori_new)
        return output_dict

    # get inputs for scene encoder
    def _get_scene_encoder_input(self, obj_pcds, scan_insts, situation = None):
        # Dict: { int: np.ndarray (N, 6) }
        if len(obj_pcds) <= self.max_obj_len:
            # Dict to List
            selected_obj_pcds = list(obj_pcds.values())
        else:
            # crop objects to max_obj_len
            selected_obj_pcds = []

            # select relevant objs first
            for i in scan_insts:
                if i in obj_pcds:
                    selected_obj_pcds.append(obj_pcds[i])

            num_selected_objs = len(selected_obj_pcds)
            if num_selected_objs >= self.max_obj_len:
                random.shuffle(selected_obj_pcds)
                selected_obj_pcds = selected_obj_pcds[:self.max_obj_len]
            else:
                # select from remaining objs
                remained_obj_idx = [i for i in obj_pcds.keys() if i not in scan_insts]
                random.shuffle(remained_obj_idx)
                for i in remained_obj_idx[: self.max_obj_len - num_selected_objs]:
                    selected_obj_pcds.append(obj_pcds[i])

            assert len(selected_obj_pcds) == self.max_obj_len
        output_dict = self.preprocess_pcd(selected_obj_pcds, return_anchor = False, rot_aug = True, situation = situation)
        return output_dict

    def __getitem__(self, index):
        one_sample = self.data[index]

        if self.modality_type == 'multimodal':
            situation = one_sample['situation_multimodal']
        else:
            situation = one_sample['situation_text']
        interaction = one_sample['interaction']
        anchor_loc = one_sample['location']
        anchor_orientation = one_sample['orientation']
        question = random.choice(NAVI_ACTION_POOL)
        question = interaction + " " + question

        # load scene data
        scan_id = one_sample['scan_id']
        obj_pcds = self.scan_data[scan_id]['obj_pcds']
        obj_pcds = {int(k): obj_pcds[k] for k in range(len(obj_pcds))}

        action_token_list = []
        action_gt_code = one_sample['action'][self.action_type][0]
        action_gt_code = self.action_mapping[self.action_type][action_gt_code]
        action_gt = ONESTEPNAVI_ACTION_SPACE_TOKENIZE[action_gt_code]
        action_token_list.append(action_gt)

        action_text_list = []
        action_text = one_sample['action'][self.action_type][1]
        action_text_list.append(action_text)

        ### scene input ####
        output_dict = self._get_scene_encoder_input(obj_pcds, one_sample['insts'], situation = (anchor_loc, anchor_orientation))
        obj_fts = output_dict['obj_fts']
        obj_locs = output_dict['obj_locs']
        anchor_loc, anchor_orientation = output_dict["situation"]
        
        data_dict = {
            "situation": situation,
            "situation_pos": np.array(anchor_loc),
            "situation_rot": np.array(anchor_orientation),
            "question": question,
            "action_token_list": action_token_list,
            "action_text_list": action_text_list,
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "scan_id": scan_id,
        }
        return data_dict