import collections
import json
import os
import random

import re
import jsonlines
import nltk
import numpy as np
import pandas as pd
import torch
import cv2

from accelerate.logging import get_logger
from einops import rearrange
from scipy import sparse
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from ..data_utils import build_rotate_mat, face_vector_in_xy_to_quaternion
from .default import DATASET_REGISTRY
from .scan_data_loader import ScanDataLoader
from .scannet import ScanNetSQA3D
from .one_step_navi import ScanNetOneStepNavi
from .text_pool import Leo_situation_pool

LLAMA_TOKEN_SENT_RATIO = 0.24

MSR3D_REQUIRED_KEYS = [
    'msr3d_prompt',
    'msr3d_imgs',  ## (B, max_num, C, H, W) this will be padded to max_num in the dataset wrapper
    'obj_fts',
    # 'obj_masks', # this is filled by dataset wrapper
    'obj_locs',
    'img_fts',
    'img_masks',
    'text_output',
    'anchor_orientation',
    'anchor_locs',
    'source',
    'scan_id',
    'prompt_before_obj',
    'prompt_middle_1',
    'prompt_middle_2',
    'prompt_after_obj',
    'index',
    'type'
]

# use a global cache to avoid loading the same data multiple times
scan_cache_data = {}

""" interleaved input format in the in-context format:
    role_prompt + situation_prompt (if existed) + scene_prompt + context prompt + task_prompt
"""
class MSR3DBase(Dataset):
    prompt_dict = {
        "role_prompt" : "You are an AI visual assistant situated in a 3D scene. ",
        "situation_prompt" : "You are at a selected location in the 3D scene. {situation}",
        "scene_prompt" : "Objects (including you) in the scene: <SCENE> ",
        "task_prompt" : "USER: {instruction} ASSISTANT:",
        "context_templete" : "USER: {Q} ASSISTANT: {A}",
    }
    place_holder_dict = {
        "IMG" : "图",
        "PCD" : "物",
        "SCENE" : "景",
    }
    prompt_combine_list = ["role_prompt", "situation_prompt", "scene_prompt", "task_prompt"]

    def __init__(self, cfg, dataset):
        self.scan_data_loader = ScanDataLoader(cfg, dataset = dataset)

    @staticmethod
    def get_text_prompts(instruction, situation=""):
        output_text = ""
        for prompt in MSR3DBase.prompt_combine_list:
            if prompt == "situation_prompt":
                output_text += MSR3DBase.prompt_dict[prompt].format(situation=situation)
            elif prompt == "task_prompt":
                output_text += MSR3DBase.prompt_dict[prompt].format(instruction=instruction)
            else:
                output_text += MSR3DBase.prompt_dict[prompt]
        return output_text
    
    @staticmethod
    def get_prompts(instruction, situation="", dialogue=None):
        return {
            'prompt_before_obj': MSR3DBase.role_prompt + MSR3DBase.situation_prompt.format(situation=situation),
            'prompt_middle_1': MSR3DBase.egoview_prompt,
            'prompt_middle_2': MSR3DBase.objects_prompt,
            'prompt_after_obj': MSR3DBase.task_prompt.format(instruction=instruction) if dialogue is None else dialogue,
        }

    """
    example place holder: <chair-1-IMG>
    """
    @staticmethod
    def parse_place_holder(text):
        pattern = r"<(.*?)>"
        matches = re.findall(pattern, text)

        # embed()
        for i, match in enumerate(matches):
            if match.split("-")[-1] in MSR3DBase.place_holder_dict:
                text = text.replace(f"<{match}>", MSR3DBase.place_holder_dict[match.split("-")[-1]])
        return text, matches

    @staticmethod
    def check_output_and_fill_dummy(data_dict):
        if 'anchor_orientation' not in data_dict:
            data_dict['anchor_orientation'] = torch.zeros(4).float()
            data_dict['anchor_orientation'][-1] = 1
        if 'anchor_locs' not in data_dict:
            data_dict['anchor_locs'] = torch.zeros(3).float()
        if 'scan_id' not in data_dict:
            data_dict['scan_id'] = ""
        if 'source' not in data_dict:
            data_dict['source'] = ""
        if 'index' not in data_dict:
            data_dict['index'] = -1
        if 'type' not in data_dict:
            data_dict['type'] = ""

        ### fill in leo output ###
        key_list = ['prompt_before_obj', 'prompt_middle_1', 'prompt_middle_2', 'prompt_after_obj']  
        for one_key in key_list:
            if one_key not in data_dict:
                data_dict[one_key] = ""

        for key in MSR3D_REQUIRED_KEYS:
            if key not in data_dict:
                raise ValueError(f'Key {key} is missing in data_dict.')
        return data_dict

    @staticmethod
    def cluster_data_with_type(data):
        cluster_data = {}
        for d in data:
            scan_id = d['scan_id']
            if scan_id not in cluster_data:
                cluster_data[scan_id] = {}
            data_type = d['type']
            if data_type not in cluster_data[scan_id]:
                cluster_data[scan_id][data_type] = []
            cluster_data[scan_id][data_type].append(d)
        return cluster_data

    def replace_all_imgs_with_txt(self, data):
        '''
        <obj-2-M> -> obj
        '''
        def replacement(match):
            obj = match.group(1)
            return obj
    
        pattern = re.compile(r"<([^<>-]+)-\d+-IMG>")
        # Replace matches in the input string according to the specified output type
        output_string = re.sub(pattern, replacement, data)
        return output_string

    ### load with global cache dict ###
    def prepare_data_loading_with_cache(self, dataset_name, scan_id, data_type_list = []):
        global scan_cache_data
        if dataset_name not in scan_cache_data:
            scan_cache_data[dataset_name] = {}

        if scan_id not in scan_cache_data[dataset_name]:
            scan_cache_data[dataset_name][scan_id] = {}

        data_type_to_process = []
        for data_type in data_type_list:
            if data_type not in scan_cache_data[dataset_name][scan_id]:
                data_type_to_process.append(data_type)
        if len(data_type_to_process) > 0:
            one_scan = self.scan_data_loader.get_data(dataset_name, scan_id, data_type = data_type_to_process)
            scan_cache_data[dataset_name][scan_id].update(one_scan)

        return scan_cache_data[dataset_name][scan_id]
    
    def preprocess_pcd(self, obj_pcds, return_anchor = False, rot_aug = True, situation = None):
        # rotate scene
        rot_matrix = build_rotate_mat(self.split, rot_aug = rot_aug)

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
                # print(f"rot_matrix: {rot_matrix}")
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

    def _split_sentence(self, sentence, max_length, prefix=''):
        # Only split during training
        if self.split == 'train' and len(prefix + sentence) > max_length:
            all_caps = []
            sents = sentence.split('. ')
            tmp = prefix
            for i in range(len(sents)):
                if len(tmp + sents[i] + '. ') > max_length:
                    all_caps.append(tmp)
                    tmp = prefix
                tmp += sents[i] + '. '

            all_caps.append(tmp)   # last chunk

            # final check
            ret = []
            for cap in all_caps:
                if len(cap) <= max_length:
                    ret.append(cap)
            return ret
        else:
            return [prefix + sentence]

    # get inputs for scene encoder
    def _get_scene_encoder_input(self, scan_data, scan_insts, situation = None):
        obj_pcds = scan_data['obj_pcds'].copy()
        # Dict: { int: np.ndarray (N, 6) }
        if len(obj_pcds) <= self.max_obj_len:
            # Dict to List
            selected_obj_pcds = list(obj_pcds.values())
        else:
            # crop objects to max_obj_len
            selected_obj_pcds = []

            # select relevant objs first
            for i in scan_insts:
                if i not in obj_pcds:
                    continue
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

        output_dict = self.preprocess_pcd(selected_obj_pcds, return_anchor = False, rot_aug = self.use_rotate, situation = situation)

        return output_dict

    @staticmethod
    def transfer_leo_to_msr3d(data_dict):
        prompt = f"{data_dict['prompt_before_obj']} {data_dict['prompt_middle_2']}{MSR3DBase.place_holder_dict['SCENE']}. {data_dict['prompt_after_obj']}"
        
        data_dict.update({
            'msr3d_prompt' : prompt,
            'msr3d_imgs': [],
        })
        return data_dict

"""
json format
{
    "scene0000_00": {
        "response" : [
            {
                "Q": "What is the color of the office chair in front of me?", 
                "A": ["gray"],            
                "query_type": "qa_4_directions", 
                "type": "attribute-color", 
                "situation": "To my left, at a middle distance, there's a gray fabric office chair with a curved rectangle shape. Far in front, there's a gray plastic bin. Far behind, there's a crumpled red pillow and a partly open grey curtain. Near to my right, there's a black and brown fabric office chair.", 
                "location": [0.08045649528503418, -0.19432830810546875, 0.026395104825496674], 
                "orientation": [0.6343910511301244, 0.7730122859605894, 0], ### face pt
                "mode": "txt"
            }
        ]
    }
}
"""
@DATASET_REGISTRY.register()
class MSQAScanNet(MSR3DBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, dataset = 'ScanNet')
        self.split = split
        self.dataset_cfg = cfg.data.msqa_scannet.args
        self.cfg = cfg

        self.num_points = self.dataset_cfg.get('num_points', 1024)
        self.max_obj_len = self.dataset_cfg.get('max_obj_len', 60)
        self.val_num = self.dataset_cfg.get('val_num', 1000)
        self.few_shot_num = self.dataset_cfg.get('few_shot_num', 0)
        self.use_rotate = self.dataset_cfg.get('use_rotate', True)
        self.use_rotate = self.use_rotate and self.split == 'train'

        print(f"Loading MSQAScanNet {split}-set language")
        self.data = self.load_lang(self.dataset_cfg.anno_dir, split)
        # scan_ids may be repeatitive
        if cfg.debug.flag:
            self.data = self.data[:cfg.debug.debug_size]
        # elif self.split == 'train':
        #     self.data = self.data[:-self.val_num]
        # else:
        #     self.data = self.data[-self.val_num:]
        self.data_dict_with_type = self.cluster_data_with_type(self.data)
        print(f"Finish loading MSQAScanNet {split}-set language, collected {len(self.data)} data")

    def load_lang(self, anno_dir, split):
        output_list = []
        fname = f'msqa_scannet_{split}.json'
        with open(os.path.join(anno_dir, fname)) as f:
            json_data = json.load(f)
        for meta_anno in json_data:
            insts = meta_anno['raw_thought'].split(', ')
            try:
                insts = [int(s.split('-')[-1]) for s in insts]
            except:
                insts = []
            meta_anno['insts'] = insts
            output_list.append(meta_anno)                      
        return output_list

    def __len__(self):
        return len(self.data)

    # get in context sampling from same scene ####
    # set to 0 in MSR3D
    def _get_context_prompt(self, one_sample, scan_id):
        data_type = one_sample['type']
        context_list = self.data_dict_with_type[scan_id][data_type]
        cur_idx = self.data_dict_with_type[scan_id][data_type].index(one_sample)
        context_idx_list = list(range(len(self.data_dict_with_type[scan_id][data_type])))
        context_idx_list.remove(cur_idx)
        context_idx_sample = random.sample(context_idx_list, min(len(context_idx_list), self.few_shot_num))
        context = ""
        for idx in context_idx_sample:
            context_question = context_list[idx]['question']
            context_answer = random.choice(context_list[idx]['answers'])
            context += MSR3DBase.prompt_dict['context_templete'].format(Q=context_question, A=context_answer)
        return context

    def replace_img_with_txt(self, data, id):
        '''
        <obj-2-M> -> obj
        '''
        def replacement(match):
            obj = match.group(1)
            return obj
    
        pattern = re.compile(fr"<([^<>-]+)-{id}-IMG>")
        # Replace matches in the input string according to the specified output type
        output_string = re.sub(pattern, replacement, data)
        return output_string

    def __getitem__(self, index):
        one_sample = self.data[index]
        question = one_sample['question']
        answer_list = one_sample['answers']
        situation = one_sample['situation']
        anchor_loc = one_sample['location']
        anchor_orientation = one_sample['orientation'] ### face pt
        qa_type = one_sample['type']
        index = one_sample['index']
        anchor_orientation = face_vector_in_xy_to_quaternion(anchor_orientation)

        scan_id = one_sample['scan_id']
        data_dict = {}

        ### prepare interleaved input prompt with in context learning
        prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
        _, place_holder_list = self.parse_place_holder(prompt)  

        ### load data with global cache ###
        scan_data = self.prepare_data_loading_with_cache(dataset_name = 'ScanNet', scan_id = scan_id, data_type_list = ['obj_pcds'])
        ### scene input ####
        output_dict = self._get_scene_encoder_input(scan_data, one_sample['insts'], situation = (anchor_loc, anchor_orientation))
        obj_fts = output_dict['obj_fts']
        obj_locs = output_dict['obj_locs']
        anchor_loc, anchor_orientation = output_dict["situation"]

        ### process place holder ####
        img_list = []
        for place_holder in place_holder_list:
            place_holder_info = place_holder.split("-")
            if place_holder_info[-1] == "SCENE":
                continue

            ### TODO: hack all place holder has the same format; remove this hack
            if len(place_holder_info) != 3:
                continue
                
            assert len(place_holder_info) == 3, print(place_holder_info) # place holder format must be <label-inst_id-type>
            cls_label, inst_id, holder_type = place_holder_info
            if holder_type == "IMG":
                # # assert inst_id in scan_data['mv_info'].keys(), print(f"inst_id {inst_id} not in scan_data['mv_info'].keys()", scan_data['mv_info'].keys())
                # if int(inst_id) not in scan_data['mv_info'].keys():
                #     print(scan_id, inst_id, scan_data['mv_info'].keys(), place_holder_info)
                #     prompt = self.replace_img_with_txt(prompt, inst_id)
                #     print(prompt)
                # else:
                #     one_bbox = random.choice(scan_data['mv_info'][int(inst_id)])
                #     img_sample = self.scan_data_loader.get_one_img(one_bbox)
                img = self.scan_data_loader.get_one_certain_img(scan_id, int(inst_id), cls_label)
                if img == None:
                    prompt = self.replace_img_with_txt(prompt, inst_id)
                else:
                    img_list.append(img)
            else:
                raise NotImplementedError(f"holder type {holder_type} not supported")
        
        if prompt.count("IMG")!=len(img_list):
            img_list = []
            prompt = self.replace_all_imgs_with_txt(prompt)

        prompt, _ = self.parse_place_holder(prompt) 
        assert(prompt.count("图")==len(img_list))

        data_dict.update({
            'source': 'msqa_scannet',
            'scan_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            # 'anchor_locs': anchor_loc.float(),
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'text_output': random.choice(answer_list),
            'answer_list': '[answer_seq]'.join(answer_list),
            # default_collate cannot collate variable-length lists, so encode to str, and decode during evaluation
            'msr3d_prompt': prompt,
            'msr3d_imgs': img_list,
            'anchor_orientation': torch.tensor(anchor_orientation).float(),
            'anchor_locs' : torch.tensor(anchor_loc).float(),
            'index': index,
            'type': qa_type
        })

        return self.check_output_and_fill_dummy(data_dict)

@DATASET_REGISTRY.register()
class SQA3DScanNet(ScanNetSQA3D, MSR3DBase):
    situation_pool = Leo_situation_pool

    def convert_person_view(self, sentence):
        forms = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'am' : 'are'}
        def translate(word):
            if word.lower() in forms:
                return forms[word.lower()]
            return word
        result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(sentence)])
        return result

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        # for instruction-following
        # data_dict_extra = self.get_prompts(
        #     instruction=data_dict['question'],
        #     situation=random.choice(self.situation_pool) + ' ' + self.convert_person_view(data_dict['situation']),
        # )
        data_dict_extra = self.get_prompts(
            instruction=data_dict['question'],
            situation=random.choice(self.situation_pool) + ' ' + self.convert_person_view(data_dict['situation']),
        )
        data_dict.update(data_dict_extra)
        data_dict.update({
            'source': 'scannet',
            'text_output': random.choice( data_dict['answer_list'].split('[answer_seq]') ),
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'anchor_locs': torch.from_numpy(data_dict['situation_pos']).float(),
            'anchor_orientation': torch.from_numpy(data_dict['situation_rot']).float(),
            'task': 'sqa3d'
        })
        data_dict = MSR3DBase.transfer_leo_to_msr3d(data_dict)

        return MSR3DBase.check_output_and_fill_dummy(data_dict)

@DATASET_REGISTRY.register()
class MSQA3RScan(MSR3DBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, dataset = '3RScan')
        self.split = split
        self.dataset_cfg = cfg.data.msqa_3rscan.args
        self.cfg = cfg

        self.num_points = self.dataset_cfg.get('num_points', 1024)
        self.max_obj_len = self.dataset_cfg.get('max_obj_len', 60)
        self.val_num = self.dataset_cfg.get('val_num', 1000)
        self.few_shot_num = self.dataset_cfg.get('few_shot_num', 0)
        self.use_rotate = self.dataset_cfg.get('use_rotate', True)
        self.use_rotate = self.use_rotate and self.split == 'train'

        print(f"Loading MSQA3RScan {split}-set language")
        self.data = self.load_lang(self.dataset_cfg.anno_dir, split)
        # scan_ids may be repeatitive
        if cfg.debug.flag:
            self.data = self.data[:cfg.debug.debug_size]
        # elif self.split == 'train':
        #     self.data = self.data[:-self.val_num]
        # else:
        #     self.data = self.data[-self.val_num:]
        self.data_dict_with_type = self.cluster_data_with_type(self.data)
        print(f"Finish loading MSQA3RScan {split}-set language, collected {len(self.data)} data")

    def load_lang(self, anno_dir, split):
        output_list = []
        fname = f'msqa_rscan_{split}.json'
        with open(os.path.join(anno_dir, fname)) as f:
            json_data = json.load(f)
        for meta_anno in json_data:
            insts = meta_anno['raw_thought'].split(', ')
            try:
                insts = [int(s.split('-')[-1]) for s in insts]
            except:
                insts = []
            meta_anno['insts'] = insts
            output_list.append(meta_anno)                      
        return output_list                  

    def __len__(self):
        return len(self.data)

    # get in context sampling from same scene ####
    def _get_context_prompt(self, one_sample, scan_id):
        data_type = one_sample['type']
        context_list = self.data_dict_with_type[scan_id][data_type]
        cur_idx = self.data_dict_with_type[scan_id][data_type].index(one_sample)
        context_idx_list = list(range(len(self.data_dict_with_type[scan_id][data_type])))
        context_idx_list.remove(cur_idx)
        context_idx_sample = random.sample(context_idx_list, min(len(context_idx_list), self.few_shot_num))
        context = ""
        for idx in context_idx_sample:
            context_question = context_list[idx]['question']
            context_answer = random.choice(context_list[idx]['answers'])
            context += MSR3DBase.prompt_dict['context_templete'].format(Q=context_question, A=context_answer)
        return context

    def replace_img_with_txt(self, data, id):
        '''
        <obj-2-M> -> obj
        '''
        def replacement(match):
            obj = match.group(1)
            return obj
    
        pattern = re.compile(fr"<([^<>-]+)-{id}-IMG>")
        # Replace matches in the input string according to the specified output type
        output_string = re.sub(pattern, replacement, data)
        return output_string

    def __getitem__(self, index):
        one_sample = self.data[index]
        question = one_sample['question']
        answer_list = one_sample['answers']
        situation = one_sample['situation']
        anchor_loc = one_sample['location']
        anchor_orientation = one_sample['orientation'] ### face pt
        qa_type = one_sample['type']
        index = one_sample['index']
        anchor_orientation = face_vector_in_xy_to_quaternion(anchor_orientation)

        scan_id = one_sample['scan_id']
        data_dict = {}

        prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
        _, place_holder_list = self.parse_place_holder(prompt)  

        ### load data with global cache ###
        scan_data = self.prepare_data_loading_with_cache(dataset_name = '3RScan', scan_id = scan_id, data_type_list = ['obj_pcds'])
        ### scene input ####
        output_dict = self._get_scene_encoder_input(scan_data, one_sample['insts'], situation = (anchor_loc, anchor_orientation))
        obj_fts = output_dict['obj_fts']
        obj_locs = output_dict['obj_locs']
        anchor_loc, anchor_orientation = output_dict["situation"]
        
        ### process place holder ####
        img_list = []
        for place_holder in place_holder_list:
            place_holder_info = place_holder.split("-")
            if place_holder_info[-1] == "SCENE":
                continue

            ### TODO: hack all place holder has the same format; remove this hack
            if len(place_holder_info) != 3:
                continue
                
            assert len(place_holder_info) == 3, print(place_holder_info) # place holder format must be <label-inst_id-type>
            cls_label, inst_id, holder_type = place_holder_info
            if holder_type == "IMG":
                # assert inst_id in scan_data['mv_info'].keys(), print(f"inst_id {inst_id} not in scan_data['mv_info'].keys()", scan_data['mv_info'].keys())
                # if int(inst_id) not in scan_data['mv_info'].keys():
                #     print(scan_id, inst_id, scan_data['mv_info'].keys(), place_holder_info)
                #     prompt = self.replace_img_with_txt(prompt, inst_id)
                #     print(prompt)
                # else:
                #     one_bbox = random.choice(scan_data['mv_info'][int(inst_id)])
                #     img_sample = self.scan_data_loader.get_one_img(one_bbox)
                img = self.scan_data_loader.get_one_certain_img(scan_id, int(inst_id), cls_label)
                if img ==  None:
                    prompt = self.replace_img_with_txt(prompt, inst_id)
                else:
                    img_list.append(img)
            else:
                raise NotImplementedError(f"holder type {holder_type} not supported")
        
        if prompt.count("IMG")!=len(img_list):
            img_list = []
            prompt = self.replace_all_imgs_with_txt(prompt)

        prompt, _ = self.parse_place_holder(prompt) 
        assert(prompt.count("图")==len(img_list))

        data_dict.update({
            'source': 'msqa_3rscan',
            'scan_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            # 'anchor_locs': anchor_loc.float(),
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'text_output': random.choice(answer_list),
            'answer_list': '[answer_seq]'.join(answer_list),
            # default_collate cannot collate variable-length lists, so encode to str, and decode during evaluation
            'msr3d_prompt': prompt,
            'msr3d_imgs': img_list,
            'anchor_orientation': torch.tensor(anchor_orientation).float(),
            'anchor_locs' : torch.tensor(anchor_loc).float(),
            'index': index,
            'type': qa_type
        })

        return self.check_output_and_fill_dummy(data_dict)

@DATASET_REGISTRY.register()
class MSQAARkitScenes(MSR3DBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, dataset = 'ARkit')
        self.split = split
        self.dataset_cfg = cfg.data.msqa_arkitscenes.args
        self.cfg = cfg

        self.num_points = self.dataset_cfg.get('num_points', 1024)
        self.max_obj_len = self.dataset_cfg.get('max_obj_len', 60)
        self.val_num = self.dataset_cfg.get('val_num', 1000)
        self.few_shot_num = self.dataset_cfg.get('few_shot_num', 0)
        self.use_rotate = self.dataset_cfg.get('use_rotate', True)
        self.use_rotate = self.use_rotate and self.split == 'train'

        print(f"Loading MSQAARkitScenes {split}-set language")
        self.data = self.load_lang(self.dataset_cfg.anno_dir, split)
        # scan_ids may be repeatitive
        if cfg.debug.flag:
            self.data = self.data[:cfg.debug.debug_size]
        # elif self.split == 'train':
        #     self.data = self.data[:-self.val_num]
        # else:
        #     self.data = self.data[-self.val_num:]
        self.data_dict_with_type = self.cluster_data_with_type(self.data)
        print(f"Finish loading MSQAARkitScenes {split}-set language, collected {len(self.data)} data")

    def load_lang(self, anno_dir, split):
        output_list = []
        fname = f'msqa_arkitscenes_{split}.json'
        with open(os.path.join(anno_dir, fname)) as f:
            json_data = json.load(f)
        for meta_anno in json_data:
            insts = meta_anno['raw_thought'].split(', ')
            try:
                insts = [int(s.split('-')[-1]) for s in insts]
            except:
                insts = []
            meta_anno['insts'] = insts
            output_list.append(meta_anno)                
        return output_list

    def __len__(self):
        return len(self.data)

    # get in context sampling from same scene ####
    def _get_context_prompt(self, one_sample, scan_id):
        data_type = one_sample['type']
        context_list = self.data_dict_with_type[scan_id][data_type]
        cur_idx = self.data_dict_with_type[scan_id][data_type].index(one_sample)
        context_idx_list = list(range(len(self.data_dict_with_type[scan_id][data_type])))
        context_idx_list.remove(cur_idx)
        context_idx_sample = random.sample(context_idx_list, min(len(context_idx_list), self.few_shot_num))
        context = ""
        for idx in context_idx_sample:
            context_question = context_list[idx]['question']
            context_answer = random.choice(context_list[idx]['answers'])
            context += MSR3DBase.prompt_dict['context_templete'].format(Q=context_question, A=context_answer)
        return context

    def replace_img_with_txt(self, data, id):
        '''
        <obj-2-M> -> obj
        '''
        def replacement(match):
            obj = match.group(1)
            return obj
    
        pattern = re.compile(fr"<([^<>-]+)-{id}-IMG>")
        # Replace matches in the input string according to the specified output type
        output_string = re.sub(pattern, replacement, data)
        return output_string


    def __getitem__(self, index):
        one_sample = self.data[index]
        question = one_sample['question']
        answer_list = one_sample['answers']
        situation = one_sample['situation']
        anchor_loc = one_sample['location']
        anchor_orientation = one_sample['orientation'] ### face pt
        qa_type = one_sample['type']
        index = one_sample['index']
        anchor_orientation = face_vector_in_xy_to_quaternion(anchor_orientation)
        qa_type = one_sample['type']

        scan_id = one_sample['scan_id']
        data_dict = {}

        prompt = MSR3DBase.get_text_prompts(instruction=question, situation=situation)
        _, place_holder_list = self.parse_place_holder(prompt)  

        ### load data with global cache ###
        scan_data = self.prepare_data_loading_with_cache(dataset_name = 'ARkit', scan_id = scan_id, data_type_list = ['obj_pcds'])
        ### scene input ####
        try:
            output_dict = self._get_scene_encoder_input(scan_data, one_sample['insts'], situation = (anchor_loc, anchor_orientation))
            obj_fts = output_dict['obj_fts']
            obj_locs = output_dict['obj_locs']
            anchor_loc, anchor_orientation = output_dict["situation"]
        except:
            print(f"!!!!!!!!!!!!!!!!scan_id: {scan_id} !!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!!!!!insts: {one_sample['insts']} !!!!!!!!!!!!!!!")
        
        ### process place holder ####
        img_list = []
        for place_holder in place_holder_list:
            place_holder_info = place_holder.split("-")
            if place_holder_info[-1] == "SCENE":
                continue

            if len(place_holder_info) != 3:
                continue
                
            assert len(place_holder_info) == 3, print(place_holder_info) # place holder format must be <label-inst_id-type>
            cls_label, inst_id, holder_type = place_holder_info
            if holder_type == "IMG":
                # assert inst_id in scan_data['mv_info'].keys(), print(f"inst_id {inst_id} not in scan_data['mv_info'].keys()", scan_data['mv_info'].keys())
                # if int(inst_id) not in scan_data['mv_info'].keys():
                #     print(scan_id, inst_id, scan_data['mv_info'].keys(), place_holder_info)
                #     prompt = self.replace_img_with_txt(prompt, inst_id)
                #     # print(prompt)
                # else:
                #     one_bbox = random.choice(scan_data['mv_info'][int(inst_id)])
                #     img_sample = self.scan_data_loader.get_one_img(one_bbox)

                img = self.scan_data_loader.get_one_certain_img(scan_id, int(inst_id), cls_label)
                if img == None:
                    prompt = self.replace_img_with_txt(prompt, inst_id)
                else:
                    img_list.append(img)
            else:
                raise NotImplementedError(f"holder type {holder_type} not supported")
        
        if prompt.count("IMG")!=len(img_list):
            img_list = []
            prompt = self.replace_all_imgs_with_txt(prompt)

        prompt, _ = self.parse_place_holder(prompt) 
        
        try:
            assert(prompt.count("图")==len(img_list))
        except:
            print(prompt)
            print(img_list)
            print(f"index: {one_sample['qa_pair']['index']}")
        
        data_dict.update({
            'source': 'msqa_arkitscenes',
            'scan_id': scan_id,
            'obj_fts': obj_fts,
            'obj_locs': obj_locs,
            # 'anchor_locs': anchor_loc.float(),
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'text_output': random.choice(answer_list),
            'answer_list': '[answer_seq]'.join(answer_list),
            # default_collate cannot collate variable-length lists, so encode to str, and decode during evaluation
            'msr3d_prompt': prompt,
            'msr3d_imgs': img_list,
            'anchor_orientation': torch.tensor(anchor_orientation).float(),
            'anchor_locs' : torch.tensor(anchor_loc).float(),
            'index': qa_type,
            'type': qa_type
        })

        return self.check_output_and_fill_dummy(data_dict)


@DATASET_REGISTRY.register()
class MSR3DMSNN(ScanNetOneStepNavi):
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        msr3d_prompt = MSR3DBase.get_text_prompts(instruction = data_dict['question'], situation = data_dict['situation'])
        msr3d_prompt, _ = MSR3DBase.parse_place_holder(msr3d_prompt)

        data_dict.update({
            'msr3d_prompt': msr3d_prompt,
            'msr3d_imgs': [],
            'text_output': random.choice(data_dict['action_token_list']),
            'source': 'scannet',
            'img_fts': torch.zeros(3, 224, 224),
            'img_masks': torch.LongTensor([0]).bool(),
            'anchor_locs': torch.from_numpy(data_dict['situation_pos']).float(),
            'anchor_orientation': torch.from_numpy(data_dict['situation_rot']).float(),
            'task': 'one_step_navi'
        })

        return MSR3DBase.check_output_and_fill_dummy(data_dict)

@DATASET_REGISTRY.register()
class MSR3DMix(Dataset):
    mapping = {
        'msqa_scannet': MSQAScanNet,
        'msqa_3rscan': MSQA3RScan,
        'msqa_arkitscenes': MSQAARkitScenes,
        'sqa3d': SQA3DScanNet,
        'scannet_one_step_navi': MSR3DMSNN,
    }

    def __init__(self, cfg, split):
        self.logger = get_logger(__name__)
        self.ratio = cfg.data.msr3dmix.args.get('ratio', 1.0)
        self.dataset_list = cfg.data.msr3dmix.args.mix
        self.datasets = []

        print(f'MSR3DMix about to load: {self.dataset_list}.', flush=True)
        for dataset in self.dataset_list:
            self.datasets.append(self.mapping[dataset](cfg, split))
            print(f'MSR3DMix finishes loading dataset: {dataset}.', flush=True)

        if type(self.ratio) == int or type(self.ratio) == float:
            self.index_range = list(np.cumsum([int(len(i)*self.ratio) for i in self.datasets]))
        else:
            self.index_range = list(np.cumsum([int(len(i)*self.ratio[ind]) for ind, i in enumerate(self.datasets)]))
        self.index_range = [0] + self.index_range
        print(f'Indecies of these datasets: {self.index_range}')

    def __len__(self):
        return self.index_range[-1]

    @staticmethod
    def streamline_output(data_dict):
        new_data_dict = {}
        for key in MSR3D_REQUIRED_KEYS:
            if key not in data_dict:
                raise ValueError(f'Key {key} is missing in data_dict.')
            else:
                new_data_dict[key] = data_dict[key]
        return new_data_dict

    def __getitem__(self, index):
        for i in range(len(self.index_range)-1):
            if self.index_range[i] <= index < self.index_range[i+1]:
                data_dict = self.datasets[i][index-self.index_range[i]]
                ### transfer leo data to msr3d format ###
                if data_dict['prompt_before_obj'] != "":
                    data_dict = MSR3DBase.transfer_leo_to_msr3d(data_dict)

                break

        return self.streamline_output(data_dict)
    
