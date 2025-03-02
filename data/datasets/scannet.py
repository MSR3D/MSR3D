import os
import collections
import json
import random
from copy import deepcopy
import jsonlines
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from .default import DATASET_REGISTRY
from ..data_utils import convert_pc_to_box, ScanQAAnswer, SQA3DAnswer, construct_bbox_corners, \
                         eval_ref_one_sample, is_explicitly_view_dependent, get_sqa_question_type, \
                         face_vector_in_xy_to_quaternion, build_rotate_mat
from .scannet_base import ScanNetBase

@DATASET_REGISTRY.register()
class ScanNetPretrain(ScanNetBase):
    def __init__(self, cfg, split, sources=None):
        super(ScanNetPretrain, self).__init__(cfg, split)
        self.pc_type = cfg.data.pretrain.args.pc_type
        self.max_obj_len = cfg.data.pretrain.args.max_obj_len
        self.num_points = cfg.data.pretrain.args.num_points
        self.scannet_scan_ids = self._load_split(cfg, split)

        if self.split == 'train':
            split_cfg = cfg.data.pretrain.args.scannet_train
        else:
            split_cfg = cfg.data.pretrain.args.scannet_val

        print(f"Loading ScanNet {split}-set language")
        self.lang_data = self._load_lang(split_cfg, sources)
        print(f"Finish loading ScanNet {split}-set language")

        print(f"Loading ScanNet {split}-set scans")
        self.scan_data = self._load_scannet(self.scannet_scan_ids, self.pc_type,
                                           load_inst_info = True)
        print(f"Finish loading ScanNet {split}-set data")

    def __getitem__(self, index):
        item = self.lang_data[index]
        dataset = item[0]
        scan_id = item[1]
        sentence = item[2]

        # scene_pcds = self.scan_data[scan_id]['scene_pcds']

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']

        # filter out background
        selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) 
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            remained_obj_idx = [i for i in range(len(obj_pcds))]
            random.shuffle(remained_obj_idx)
            selected_obj_idxs = remained_obj_idx[:self.max_obj_len]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            assert len(obj_pcds) == self.max_obj_len

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels)

        data_dict = {'source': dataset,
                     'scan_id': scan_id,
                     'sentence': sentence,
                     # 'scene_pcds': scene_pcds,
                     'obj_fts': obj_fts,
                     'obj_locs': obj_locs,
                     'obj_labels': obj_labels} 
        return data_dict

@DATASET_REGISTRY.register()
class ScanNetMVPretrain(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetMVPretrain, self).__init__(cfg, split)
        self.pc_type = cfg.data.mvdatasettings.pc_type
        self.max_inst_per_frame = cfg.data.mvdatasettings.max_inst_per_frame
        self.max_frame_num = cfg.data.mvdatasettings.max_frame_num
        self.scannet_scan_ids = self._load_split(cfg, split)

        print(f"Loading ScanNet Multiview Pretrain {split}-set language")
        self.lang_data = self._load_lang(cfg.data.mvpretrain.scan_caption)
        print(f"Finish loading ScanNet {split}-set language")

        print(f"Loading ScanNet Multiview Pretrain {split}-set scans")
        self.scan_data = self._load_scannet(self.scannet_scan_ids, self.pc_type, load_pc_info = True,
                                           load_inst_info = True, load_multiview_info = True,
                                           use_multi_process = cfg.data.mvdatasettings.use_multi_process, 
                                           process_num = cfg.data.mvdatasettings.process_num)
        print(f"Finish loading ScanNet Multiview Pretrain {split}-set data")

    def __getitem__(self, index):
        item = self.lang_data[index]
        dataset = item[0]
        scan_id = item[1]
        sentence = item[2]

        mv_out_dict = self._get_multiview_info(scan_id)

        data_dict = {
            'source': dataset,
            'scan_id': scan_id,
            'sentence': sentence,
        }

        if self.cfg.data.mvdatasettings.is_pool_obj_feature:
            data_dict['vis_obj_feats'] = []
            data_dict['vis_obj_locs'] = []
            data_dict['vis_obj_labels'] = []
            for key in mv_out_dict.keys():
                one_obj = mv_out_dict[key]
                data_dict['vis_obj_feats'].append(one_obj['feat'])
                data_dict['vis_obj_locs'].append(one_obj['location'])
                data_dict['vis_obj_labels'].append(one_obj['label'])

            # data_dict['vis_obj_feats'] = mv_out_dict['vis_obj_feats']
            # data_dict['vis_obj_locs'] = mv_out_dict['vis_obj_locs']
            # data_dict['vis_obj_labels'] = mv_out_dict['vis_obj_labels']
        else:
            data_dict['mv_inst_feats'] = mv_out_dict['mv_inst_feats'],
            data_dict['mv_inst_masks'] = mv_out_dict['mv_inst_masks'],
            data_dict['mv_inst_locs'] = mv_out_dict['mv_inst_locs'],
            data_dict['mv_camera_pose'] = mv_out_dict['mv_camera_pose'],
            data_dict['mv_inst_labels'] = mv_out_dict['mv_inst_labels'],

        return data_dict
    
@DATASET_REGISTRY.register()
class ScanNetMVReferit3D(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetMVReferit3D, self).__init__(cfg, split)

        self.data_setting = cfg.data.mvdatasettings
        self.training_setting = cfg.data.mvreferit3d.args
        self.max_obj_len = self.training_setting.max_obj_len
        self.pc_type = self.data_setting.pc_type
        assert self.pc_type in ['gt', 'pred']
        assert self.training_setting.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        assert self.training_setting.anno_type in ['nr3d', 'sr3d']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        split_scan_ids = self._load_split(cfg, self.split)

        print(f"Loading ScanNet MVReferit3D {split}-set language")
        self.lang_data, self.scan_ids = self._load_lang(split_scan_ids)
        print(f"Finish loading ScanNet MVReferit3D {split}-set language")

        print(f"Loading ScanNet MVReferit3D {split}-set scans")
        self.scan_data = self._load_scannet(split_scan_ids, self.pc_type, load_pc_info = True,
                                           load_inst_info = (self.split!='test'), load_multiview_info = True,
                                           use_multi_process = cfg.data.mvdatasettings.use_multi_process, 
                                           process_num = cfg.data.mvdatasettings.process_num)
        print(f"Finish loading MVScanNet Referit3D {split}-set data")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['tokens'])

        mv_out_dict = self._get_multiview_info(scan_id)

        inst_id_list = mv_out_dict.keys() # N
        # filter out background or language
        if self.training_setting.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = []
                for inst_id in inst_id_list:
                    obj_label =  mv_out_dict[inst_id]['label']
                    if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']) and (self.int2cat[obj_label] in sentence): 
                        selected_obj_idxs.append(inst_id)
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = []
                for inst_id in inst_id_list:
                    obj_label =  mv_out_dict[inst_id]['label']
                    if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']): 
                        selected_obj_idxs.append(inst_id)

        # build tgt object id and box
        if self.pc_type == 'gt':
            assert tgt_object_id in inst_id_list, print(tgt_object_id, inst_id_list)
            tgt_object_label = mv_out_dict[tgt_object_id]['label']
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
            assert self.int2cat[tgt_object_label] == tgt_object_name

        # crop objects
        if self.max_obj_len < len(inst_id_list):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj in inst_id_list:
                klabel = mv_out_dict[kobj]['label']
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break

            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]

        if tgt_object_id != -1:
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
        tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                    for id in tgt_object_id_iou25_list]
        tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                    for id in tgt_object_id_iou50_list]

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(selected_obj_idxs)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(selected_obj_idxs)).long()
        tgt_object_id_iou50 = torch.zeros(len(selected_obj_idxs)).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard
        }

        obj_fts = []
        obj_locs = []
        obj_labels = []
        obj_boxes = []
        for idx in range(len(selected_obj_idxs)):
            one_obj = mv_out_dict[selected_obj_idxs[idx]]
            obj_fts.append(one_obj['feat'])
            obj_locs.append(one_obj['location'])
            obj_labels.append(one_obj['label'])
            obj_boxes.append(one_obj['box'])
        
        data_dict['obj_fts'] = torch.from_numpy(np.array(obj_fts))
        data_dict['obj_locs'] = torch.from_numpy(np.array(obj_locs))
        data_dict['obj_labels'] = torch.from_numpy(np.array(obj_labels))
        data_dict['obj_boxes'] = torch.from_numpy(np.array(obj_boxes))

        assert data_dict['obj_labels'][data_dict['tgt_object_id']] == data_dict['tgt_object_label']

        return data_dict

    def _load_lang(self, split_scan_ids=None):
        lang_data = []
        scan_ids = set()

        anno_file = os.path.join(self.base_dir, f'annotations/refer/{self.training_setting.anno_type}.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                    scan_ids.add(item['scan_id'])
                    lang_data.append(item)

        if self.training_setting.sr3d_plus_aug:
            anno_file = os.path.join(self.base_dir, 'annotations/refer/sr3d+.jsonl')
            with jsonlines.open(anno_file, 'r') as _f:
                for item in _f:
                    if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                        scan_ids.add(item['scan_id'])
                        lang_data.append(item)

        #### hack rm no inst 
        rm_list = [['scene0227_00', 3], ['scene0072_02', 4]]
        out_lang_data = []
        for item in lang_data:
            is_need_rm = False
            for x in rm_list:
                if item['scan_id'] == x[0] and int(item['target_id']) == x[1]:
                    is_need_rm = True
            if not is_need_rm:
                out_lang_data.append(item)

        return out_lang_data, scan_ids


@DATASET_REGISTRY.register()
class ScanNetScanRefer(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetScanRefer, self).__init__(cfg, split)
        
        self.pc_type = cfg.data.scanrefer.args.pc_type
        self.sem_type = cfg.data.scanrefer.args.sem_type
        self.max_obj_len = cfg.data.scanrefer.args.max_obj_len - 1
        self.num_points = cfg.data.scanrefer.args.num_points
        self.filter_lang = cfg.data.scanrefer.args.filter_lang
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        split_scan_ids = self._load_split(cfg, self.split)
        
        print(f"Loading ScanNet ScanRefer {split}-set language")
        self.lang_data, self.scan_ids, self.scan_to_item_idxs = self._load_lang(split_scan_ids)
        print(f"Finish loading ScanNet ScanRefer {split}-set language")

        print(f"Loading ScanNet ScanRefer {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=self.split!='test')
        print(f"Finish loading ScanNet ScanRefer {split}-set data")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
            gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_center = self.scan_data[scan_id]['obj_center']
            obj_box_size = self.scan_data[scan_id]['obj_box_size']
            obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
            obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
            for i, _ in enumerate(obj_center_pred):
                for j, _ in enumerate(obj_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                  obj_box_size[j]),
                                           construct_bbox_corners(obj_center_pred[i],
                                                                  obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break

        # filter out background or language
        # do not filter for predicted labels, because these labels are not accurate
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_label = obj_labels[tgt_object_id]
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
            assert self.int2cat[tgt_object_label] == tgt_object_name
        elif self.pc_type == 'pred':
            gt_pcd = self.scan_data[scan_id]["pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count'][self.label_converter.id_to_scannetid
                                                             [tgt_object_label]] > 1

        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': is_multiple
        }

        return data_dict

    def _load_lang(self, split_scan_ids=None):
        lang_data = []
        scan_ids = set()
        scan_to_item_idxs = collections.defaultdict(list)

        anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids:
                    scan_ids.add(item['scan_id'])
                    scan_to_item_idxs[item['scan_id']].append(len(lang_data))
                    lang_data.append(item)

        return lang_data, scan_ids, scan_to_item_idxs

@DATASET_REGISTRY.register()
class ScanNetReferit3D(ScanNetBase):
    def __init__(self, cfg, split):
        super(ScanNetReferit3D, self).__init__(cfg, split)

        self.pc_type = cfg.data.referit3d.args.pc_type
        self.sem_type = cfg.data.referit3d.args.sem_type
        self.max_obj_len = cfg.data.referit3d.args.max_obj_len - 1
        self.num_points = cfg.data.referit3d.args.num_points
        self.filter_lang = cfg.data.referit3d.args.filter_lang
        self.anno_type = cfg.data.referit3d.args.anno_type
        self.sr3d_plus_aug = cfg.data.referit3d.args.sr3d_plus_aug
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        assert self.anno_type in ['nr3d', 'sr3d']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val
        if self.split == 'test':
            self.split = 'val'

        split_scan_ids = self._load_split(cfg, self.split)

        print(f"Loading ScanNet Referit3D {split}-set language")
        self.lang_data, self.scan_ids = self._load_lang(split_scan_ids)
        print(f"Finish loading ScanNet Referit3D {split}-set language")

        print(f"Loading ScanNet Referit3D {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=self.split!='test')
        print(f"Finish loading ScanNet Referit3D {split}-set data")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['tokens'])

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
            gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_center = self.scan_data[scan_id]['obj_center']
            obj_box_size = self.scan_data[scan_id]['obj_box_size']
            obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
            obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
            for i, _ in enumerate(obj_center_pred):
                for j, _ in enumerate(obj_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                  obj_box_size[j]),
                                           construct_bbox_corners(obj_center_pred[i],
                                                                  obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break

        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_label = obj_labels[tgt_object_id]
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
            assert self.int2cat[tgt_object_label] == tgt_object_name
        elif self.pc_type == 'pred':
            gt_pcd = self.scan_data[scan_id]["pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard
        }

        return data_dict

    def _load_lang(self, split_scan_ids=None):
        lang_data = []
        scan_ids = set()

        anno_file = os.path.join(self.base_dir, f'annotations/refer/{self.anno_type}.jsonl')
        with jsonlines.open(anno_file, 'r') as _f:
            for item in _f:
                if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                    scan_ids.add(item['scan_id'])
                    lang_data.append(item)

        if self.sr3d_plus_aug:
            anno_file = os.path.join(self.base_dir, 'annotations/refer/sr3d+.jsonl')
            with jsonlines.open(anno_file, 'r') as _f:
                for item in _f:
                    if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                        scan_ids.add(item['scan_id'])
                        lang_data.append(item)

        return lang_data, scan_ids

@DATASET_REGISTRY.register()
class ScanNetScanQA(ScanNetBase):
    def __init__(self, cfg, split, sources=None):
        super(ScanNetScanQA, self).__init__(cfg, split)

        self.pc_type = cfg.data.scanqa.args.pc_type
        self.sem_type = cfg.data.scanqa.args.sem_type
        self.max_obj_len = cfg.data.scanqa.args.max_obj_len - 1
        self.num_points = cfg.data.scanqa.args.num_points
        self.filter_lang = cfg.data.scanqa.args.filter_lang
        self.use_unanswer = cfg.data.scanqa.args.use_unanswer
        
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        
        # use validation set for evaluation
        if self.split == 'test':
            self.split = 'val'
        self.is_test = self.split == 'test'

        print(f"Loading ScanNet ScanQA {split}-set language")
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()
        lang_data, self.scan_ids, self.scan_to_item_idxs = self._load_lang()
        if cfg.debug.flag:
            self.lang_data = []
            self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            for item in lang_data:
                if item['scene_id'] in self.scan_ids:
                    self.lang_data.append(item)
        else:
            self.lang_data = lang_data
        print(f"Finish loading ScanNet ScanQA {split}-set language")

        print(f"Loading ScanNet ScanQA {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=self.split!='test')
        print(f"Finish loading ScanNet ScanQA {split}-set data")

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['question_id']
        item_id = ''.join([i for i in item_id if i.isdigit()])
        item_id = int( item_id[:-1].lstrip('0') + item_id[-1] )
        scan_id = item['scene_id']
        if not self.is_test:
            tgt_object_id_list = item['object_ids']
            tgt_object_name_list = item['object_names']
            answer_list = item['answers']
            answer_id_list = [self.answer_vocab.stoi(answer) 
                              for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]
        else:
            tgt_object_id_list = []
            tgt_object_name_list = []
            answer_list = []
            answer_id_list = []
        question = item['question']

         # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
            if not self.is_test:
                gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
                obj_center = self.scan_data[scan_id]['obj_center']
                obj_box_size = self.scan_data[scan_id]['obj_box_size']
                obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
                obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
                for i, _ in enumerate(obj_center_pred):
                    for j, _ in enumerate(obj_center):
                        if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                      obj_box_size[j]),
                                            construct_bbox_corners(obj_center_pred[i],
                                                                   obj_box_size_pred[i])) >= 0.25:
                            obj_labels[i] = gt_obj_labels[j]
                            break

        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                    if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                    and (self.int2cat[obj_label] in question)]
                for _id in tgt_object_id_list:
                    if _id not in selected_obj_idxs:
                        selected_obj_idxs.append(_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id_list = [selected_obj_idxs.index(x) for x in tgt_object_id_list]
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
            for i, _ in enumerate(tgt_object_label_list):
                assert self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i] 
        elif self.pc_type == 'pred':
            # build gt box
            gt_center = []
            gt_box_size = []
            for cur_id in tgt_object_id_list:
                gt_pcd = self.scan_data[scan_id]["obj_pcds"][cur_id]
                center, box_size = convert_pc_to_box(gt_pcd)
                gt_center.append(center)
                gt_box_size.append(box_size)

            # start filtering
            tgt_object_id_list = []
            tgt_object_label_list = []
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                for j, _ in enumerate(gt_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center,
                                                                  obj_box_size),
                                           construct_bbox_corners(gt_center[j],
                                                                  gt_box_size[j])) >= 0.25:
                        tgt_object_id_list.append(i)
                        tgt_object_label_list.append(self.cat2int[tgt_object_name_list[j]])
                        break
        assert(len(obj_pcds) == len(obj_labels))

        # crop objects
        if self.max_obj_len < len(obj_labels):
            selected_obj_idxs = tgt_object_id_list.copy()
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in  tgt_object_id_list:
                    if klabel in tgt_object_label_list:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            tgt_object_id_list = [i for i in range(len(tgt_object_id_list))]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if len(tgt_object_id_list) == 0:
            tgt_object_id_list.append(len(obj_pcds))
            tgt_object_label_list.append(5)

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True)

        # convert answer format
        answer_label = torch.zeros(self.num_answers).long()
        for _id in answer_id_list:
            answer_label[_id] = 1
        # tgt object id
        tgt_object_id = torch.zeros(len(obj_fts) + 1).long() # add 1 for pad place holder
        for _id in tgt_object_id_list:
            tgt_object_id[_id] = 1
        # tgt object sematic
        if self.sem_type == '607':
            tgt_object_label = torch.zeros(607).long()
        else:
            raise NotImplementedError("semantic type " + self.sem_type) 
        for _id in tgt_object_label_list:
            tgt_object_label[_id] = 1

        data_dict = {
            "sentence": question,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answer_list": "[answer_seq]".join(answer_list),
            "answer_label": answer_label, # A
            "tgt_object_id": torch.LongTensor(tgt_object_id), # N
            "tgt_object_label": torch.LongTensor(tgt_object_label), # L
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_labels": obj_labels,
            "obj_boxes": obj_boxes, # N, 6
            "data_idx": item_id
        }

        return data_dict

    def _load_lang(self):
        lang_data = []
        scan_ids = set()
        scan_to_item_idxs = collections.defaultdict(list)

        anno_file = os.path.join(self.base_dir,
                                 f'annotations/qa/ScanQA_v1.0_{self.split}.json')

        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                scan_to_item_idxs[item['scene_id']].append(len(lang_data))
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')
        return lang_data, scan_ids, scan_to_item_idxs

    def build_answer(self):
        train_data = json.load(open(os.path.join(self.base_dir,
                                'annotations/qa/ScanQA_v1.0_train.json'), encoding='utf-8'))
        answer_counter = sum([data['answers'] for data in train_data], [])
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = ScanQAAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands


@DATASET_REGISTRY.register()
class ScanNetScanQAInstruction(ScanNetScanQA):
    r"""Prompt format:
    <holistic prompt> Here are the object tokens in the scene: <obj_1>, <obj_2>, …, <obj_N>. Question: <question> Answer: 
    """
    holistic_prompt = "Assume you are an AI visual assistant situated in a 3D scene. You receive a sequence of object tokens in the scene, each representing the feature of a corresponding object. Next you will receive a question to answer based on the visual information embedded in the object tokens."
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        data_dict['prompt_before_obj'] = f"{self.holistic_prompt} Here are the object tokens in the scene: "
        data_dict['prompt_after_obj'] = f". Question: {data_dict['sentence']} Answer: "
        
        answer_list = data_dict['answers'].split('[answer_seq]')
        # for training, random select an answer
        data_dict['text_output'] = random.choice(answer_list)   
        return data_dict


@DATASET_REGISTRY.register()
class ScanNetSQA3D(ScanNetBase):
    r"""
    questions json file: dict_keys(['info', 'license', 'data_type', 'data_subtype', 'task_type', 'questions'])
        'questions': List
        'questions'[0]: {
            'scene_id': 'scene0050_00',
            'situation': 'I am standing by the ottoman on my right facing a couple of toolboxes.',
            'alternative_situation': [
                'I just placed two backpacks on the ottoman on my right side before I went to play the piano in front of me to the right.',
                'I stood up from the ottoman and walked over to the piano ahead of me.'
            ],
            'question': 'What instrument in front of me is ebony and ivory?',
            'question_id': 220602000002
        }

    annotations json file: dict_keys(['info', 'license', 'data_type', 'data_subtype', 'annotations'])
        'annotations': List
        'annotations'[0]: {
            'scene_id': 'scene0050_00',
            'question_type': 'N/A',
            'answer_type': 'other',
            'question_id': 220602000002,
            'answers': [{'answer': 'piano', 'answer_confidence': 'yes', 'answer_id': 1}],
            'rotation': {'_x': 0, '_y': 0, '_z': -0.9995736030415032, '_w': -0.02919952230128897},
            'position': {'x': 0.7110268899979686, 'y': -0.03219739162793617, 'z': 0}
        }
    """
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

        self.pc_type = cfg.data.sqa3d.args.pc_type
        self.sem_type = cfg.data.sqa3d.args.sem_type
        self.max_obj_len = cfg.data.sqa3d.args.max_obj_len - 1
        self.num_points = cfg.data.sqa3d.args.num_points
        self.filter_lang = cfg.data.sqa3d.args.filter_lang
        self.use_unanswer = cfg.data.sqa3d.args.use_unanswer

        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        
        # use test set for validation
        # elif self.split == 'val':
        #     self.split = 'test'
        
        print(f"Loading ScanNet SQA3D {split}-set language")

        # build answer
        self.num_answers, self.answer_vocab, self.answer_cands = self.build_answer()   
        
        # load annotations
        lang_data, self.scan_ids, self.scan_to_item_idxs = self._load_lang()
        if cfg.debug.flag:
            self.lang_data = lang_data[:cfg.debug.debug_size]
            # self.lang_data = []
            # self.scan_ids = sorted(list(self.scan_ids))[:cfg.debug.debug_size]
            # for item in lang_data:
            #     if item['scene_id'] in self.scan_ids:
            #         self.lang_data.append(item)
        else:
            self.lang_data = lang_data
        
        # load question engine
        self.questions_map = self._load_question()

        print(f"Finish loading ScanNet SQA3D {split}-set language")
        
        # load scans
        print(f"Loading ScanNet SQA3D {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type, self.pc_type == 'gt')
        print(f"Finish loading ScanNet SQA3D {split}-set data")
    
    def __getitem__(self, index):
        item = self.lang_data[index]
        item_id = item['question_id']
        scan_id = item['scene_id']

        tgt_object_id_list = []
        tgt_object_name_list = []
        answer_list = [answer['answer'] for answer in item['answers']]
        answer_id_list = [self.answer_vocab.stoi(answer) for answer in answer_list if self.answer_vocab.stoi(answer) >= 0]

        if self.split == 'train':
            # augment with random situation for train
            situation = random.choice(self.questions_map[scan_id][item_id]['situation'])
        else:
            # fix for eval
            situation = self.questions_map[scan_id][item_id]['situation'][0]
        question = self.questions_map[scan_id][item_id]['question']
        concat_sentence = situation + question
        question_type = get_sqa_question_type(question)
       
        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            
        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling']) and (self.int2cat[obj_label] in concat_sentence)]
                for _id in tgt_object_id_list:
                    if _id not in selected_obj_idxs:
                        selected_obj_idxs.append(_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels) if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]
        
        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id_list = [selected_obj_idxs.index(x) for x in tgt_object_id_list]
            tgt_object_label_list = [obj_labels[x] for x in tgt_object_id_list]
            for i in range(len(tgt_object_label_list)):
                assert self.int2cat[tgt_object_label_list[i]] == tgt_object_name_list[i]
        elif self.pc_type == 'pred':
            # build gt box
            gt_center = []
            gt_box_size = []
            for cur_id in tgt_object_id_list:
                gt_pcd = self.scan_data[scan_id]["obj_pcds"][cur_id]
                center, box_size = convert_pc_to_box(gt_pcd)
                gt_center.append(center)
                gt_box_size.append(box_size)
            
            # start filtering
            tgt_object_id_list = []
            tgt_object_label_list = []
            for i in range(len(obj_pcds)):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                for j in range(len(gt_center)):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size), construct_bbox_corners(gt_center[j], gt_box_size[j])) >= 0.25:
                        tgt_object_id_list.append(i)
                        tgt_object_label_list.append(self.cat2int[tgt_object_name_list[j]])
                        break
        assert(len(obj_pcds) == len(obj_labels))

        # crop objects
        if self.max_obj_len < len(obj_labels):
            selected_obj_idxs = tgt_object_id_list.copy()
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in  tgt_object_id_list:
                    if klabel in tgt_object_label_list:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            tgt_object_id_list = [i for i in range(len(tgt_object_id_list))]
            assert len(obj_pcds) == self.max_obj_len
        
        # rebuild tgt_object_id
        if len(tgt_object_id_list) == 0:
            tgt_object_id_list.append(len(obj_pcds))
            tgt_object_label_list.append(5)

        pcd_data = torch.load(os.path.join(self.base_dir, 'scan_data', 'pcd_with_global_alignment', f'{scan_id}.pth'))
        points = pcd_data[0]
        scene_center = (points.max(0) + points.min(0)) / 2
        pos = item['position']
        ori = item['rotation']
        pos, ori = self.transform_situation(scan_id, scene_center, pos, ori)

        obj_fts, obj_locs, obj_boxes, obj_labels, (pos, ori) = self._obj_processing_post(
            obj_pcds, obj_labels, is_need_bbox=True, situation=(pos, ori)
        )
        
        # convert answer format
        answer_label = torch.zeros(self.num_answers).long()
        for _id in answer_id_list:
            answer_label[_id] = 1
        # tgt object id
        tgt_object_id = torch.zeros(len(obj_fts) + 1).long() # add 1 for pad place holder
        for _id in tgt_object_id_list:
            tgt_object_id[_id] = 1
        # tgt object sematic
        if self.sem_type == '607':
            tgt_object_label = torch.zeros(607).long()
        else:
            raise NotImplementedError("semantic type " + self.sem_type) 
        for _id in tgt_object_label_list:
            tgt_object_label[_id] = 1
        
        data_dict = {
            "situation": situation,
            "situation_pos": pos,
            "situation_rot": ori,
            "question": question,
            "sentence": concat_sentence,
            "scan_dir": os.path.join(self.base_dir, 'scans'),
            "scan_id": scan_id,
            "answer_list": "[answer_seq]".join(answer_list),
            "answer_label": answer_label, # A
            "tgt_object_id": torch.LongTensor(tgt_object_id), # N
            "tgt_object_label": torch.LongTensor(tgt_object_label), # L
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "obj_labels": obj_labels,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "sqa_type": question_type
        }

        return data_dict

    def transform_situation(self, scene_id, scene_center, pos, ori):
        """
        Since the position and orientation are based on the original mesh, we need to transform them to the aligned point cloud
        pos: [x, y, z]
        ori: [_x, _y, _z, _w]
        """
        if isinstance(pos, dict):
            pos = [pos['x'], pos['y'], pos['z']]
        pos = np.array(pos)

        if isinstance(ori, dict):
            ori = [ori['_x'], ori['_y'], ori['_z'], ori['_w']]
        ori = np.array(ori)

        with open(os.path.join(self.base_dir, f'scans/{scene_id}/{scene_id}.txt'), 'r') as f:
            scene_info = f.readlines()
        for sinfo in scene_info:
            if 'axisAlignment' in sinfo:
                values = sinfo.split('=')[1].strip().split()
                break
        assert values is not None and len(values) == 16
        rotmatrix_elements = [float(val) for val in values]
        rotmatrix = []
        for i in range(0, len(rotmatrix_elements), 4):
            rotmatrix.append(rotmatrix_elements[i:i+4])
        rotmatrix = np.array(rotmatrix)

        pos_new = pos.reshape(1, 3) @ rotmatrix[:3, :3].T
        pos_new += scene_center
        pos_new = pos_new.reshape(-1)

        ori = R.from_quat(ori).as_matrix()
        ori_new = rotmatrix[:3, :3] @ ori
        ori_new = R.from_matrix(ori_new).as_quat()
        ori_new = ori_new.reshape(-1)
        # orientation = np.array([-1, 0, 0]).reshape(1, 3) @ rotmatrix[:3, :3].T
        return pos_new, ori_new

    def build_answer(self):
        answer_data = json.load(open(os.path.join(self.base_dir, 'annotations/sqa_task/answer_dict.json')))[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        num_answers = len(answer_counter)
        answer_cands = answer_counter.keys()
        answer_vocab = SQA3DAnswer(answer_cands)
        print(f"total answers is {num_answers}")
        return num_answers, answer_vocab, answer_cands

    def _load_lang(self):
        lang_data = []
        scan_ids = set()
        scan_to_item_idxs = collections.defaultdict(list)

        anno_file = os.path.join(self.base_dir, f'annotations/sqa_task/balanced/v1_balanced_sqa_annotations_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['annotations']
        for item in json_data:
            if self.use_unanswer or (len(set(item['answers']) & set(self.answer_cands)) > 0):
                scan_ids.add(item['scene_id'])
                scan_to_item_idxs[item['scene_id']].append(len(lang_data))
                lang_data.append(item)
        print(f'{self.split} unanswerable question {len(json_data) - len(lang_data)},'
              + f'answerable question {len(lang_data)}')

        return lang_data, scan_ids, scan_to_item_idxs
    
    def _load_question(self):
        questions_map = {}
        anno_file = os.path.join(self.base_dir, f'annotations/sqa_task/balanced/v1_balanced_questions_{self.split}_scannetv2.json')
        json_data = json.load(open(anno_file, 'r', encoding='utf-8'))['questions']
        for item in json_data:
            if item['scene_id'] not in questions_map.keys():
                questions_map[item['scene_id']] = {}
            questions_map[item['scene_id']][item['question_id']] = {
                'situation': [item['situation']] + item['alternative_situation'],   # list of sentences
                'question': item['question']   # sentence
            }
        
        return questions_map


@DATASET_REGISTRY.register()
class ScanNetSQA3DInstruction(ScanNetSQA3D):
    r""" Adapted for instruction following format
    Prompt format:
    <holistic prompt> Here are the object tokens in the scene: <obj_1>, <obj_2>, …, <obj_N>. Situation: <situation> Question: <question> Answer: 
    """
    holistic_prompt = "Assume you are an AI visual assistant situated in a 3D scene. You receive a sequence of object tokens in the scene, each representing the feature of a corresponding object. And you receive a situation specifying where you are in the 3D scene. Next you will receive a question to answer based on the visual information embedded in the object tokens."
    
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)

        # for instruction-following
        data_dict.update({
            'prompt_before_obj': f"{self.holistic_prompt} Here are the object tokens in the scene: ",
            'prompt_after_obj': f". Situation: {data_dict['situation']} Question: {data_dict['question']} Answer: ",
            'text_output': random.choice( data_dict['answers'].split('[answer_seq]') )
        })
        
        return data_dict


@DATASET_REGISTRY.register()
class ScanNetSpatialRefer(ScanNetBase):
    def __init__(self, cfg, split, sources=None):
        super(ScanNetSpatialRefer, self).__init__(cfg, split)

        self.pc_type = cfg.data.spatialrefer.args.pc_type
        self.sem_type = cfg.data.spatialrefer.args.sem_type
        self.max_obj_len = cfg.data.spatialrefer.args.max_obj_len - 1
        self.num_points = cfg.data.spatialrefer.args.num_points
        self.filter_lang = cfg.data.spatialrefer.args.filter_lang

        assert sources
        assert self.pc_type in ['gt', 'pred']
        assert self.sem_type in ['607']
        assert self.split in ['train', 'val', 'test']
        # assert self.anno_type in ['nr3d', 'sr3d']
        if self.split == 'train':
            self.pc_type = 'gt'
        # TODO: hack test split to be the same as val, ScanRefer and Referit3D Diff
        if self.split == 'test':
            self.split = 'val'

        split_scan_ids = self._load_split(cfg, self.split)

        print(f"Loading ScanNet SpatialRefer {split}-set language")
        if self.split == 'train':
            split_cfg = cfg.data.spatialrefer.args.scannet_train
        else:
            split_cfg = cfg.data.spatialrefer.args.scannet_val
        self.lang_data, self.scan_ids = self._load_lang(split_cfg, sources,
                                                        split_scan_ids)
        print(f"Finish loading ScanNet SpatialRefer {split}-set language")

        print(f"Loading ScanNet SpatialRefer {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type,
                                           load_inst_info=self.split!='test')
        print(f"Finish loading ScanNet SpatialRefer {split}-set data")

         # build unique multiple look up
        for scan_id in self.scan_ids:
            inst_labels = self.scan_data[scan_id]['inst_labels']
            self.scan_data[scan_id]['label_count'] = collections.Counter(
                                    [self.label_converter.id_to_scannetid[l] for l in inst_labels])

    def __getitem__(self, index):
        """Data dict post-processing, for example, filtering, crop, nomalization,
        rotation, etc.
        Args:
            index (int): _description_
        """
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id =  item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        is_view_dependent = is_explicitly_view_dependent(item['utterance'].split(' '))

        # load pcds and labels
        if self.pc_type == 'gt':
            obj_pcds = self.scan_data[scan_id]['obj_pcds'] # N, 6
            obj_labels = self.scan_data[scan_id]['inst_labels'] # N
        elif self.pc_type == 'pred':
            obj_pcds = self.scan_data[scan_id]['obj_pcds_pred']
            obj_labels = self.scan_data[scan_id]['inst_labels_pred']
            # get obj labels by matching
            gt_obj_labels = self.scan_data[scan_id]['inst_labels'] # N
            obj_center = self.scan_data[scan_id]['obj_center']
            obj_box_size = self.scan_data[scan_id]['obj_box_size']
            obj_center_pred = self.scan_data[scan_id]['obj_center_pred']
            obj_box_size_pred = self.scan_data[scan_id]['obj_box_size_pred']
            for i, _ in enumerate(obj_center_pred):
                for j, _ in enumerate(obj_center):
                    if eval_ref_one_sample(construct_bbox_corners(obj_center[j],
                                                                  obj_box_size[j]),
                                           construct_bbox_corners(obj_center_pred[i],
                                                                  obj_box_size_pred[i])) >= 0.25:
                        obj_labels[i] = gt_obj_labels[j]
                        break

        # filter out background or language
        if self.filter_lang:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])
                                and (self.int2cat[obj_label] in sentence)]
                if tgt_object_id not in selected_obj_idxs:
                    selected_obj_idxs.append(tgt_object_id)
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]
        else:
            if self.pc_type == 'gt':
                selected_obj_idxs = [i for i, obj_label in enumerate(obj_labels)
                                if (self.int2cat[obj_label] not in ['wall', 'floor', 'ceiling'])]
            else:
                selected_obj_idxs = [i for i in range(len(obj_pcds))]

        obj_pcds = [obj_pcds[id] for id in selected_obj_idxs]
        obj_labels = [obj_labels[id] for id in selected_obj_idxs]

        # build tgt object id and box
        if self.pc_type == 'gt':
            tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_label = obj_labels[tgt_object_id]
            tgt_object_id_iou25_list = [tgt_object_id]
            tgt_object_id_iou50_list = [tgt_object_id]
            assert self.int2cat[tgt_object_label] == tgt_object_name
        elif self.pc_type == 'pred':
            gt_pcd = self.scan_data[scan_id]["pcds"][tgt_object_id]
            gt_center, gt_box_size = convert_pc_to_box(gt_pcd)
            tgt_object_id = -1
            tgt_object_id_iou25_list = []
            tgt_object_id_iou50_list = []
            tgt_object_label = self.cat2int[tgt_object_name]
            # find tgt iou 25
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.25:
                    tgt_object_id = i
                    tgt_object_id_iou25_list.append(i)
            # find tgt iou 50
            for i, _ in enumerate(obj_pcds):
                obj_center, obj_box_size = convert_pc_to_box(obj_pcds[i])
                if eval_ref_one_sample(construct_bbox_corners(obj_center, obj_box_size),
                                       construct_bbox_corners(gt_center, gt_box_size)) >= 0.5:
                    tgt_object_id_iou50_list.append(i)
        assert len(obj_pcds) == len(obj_labels)

        # crop objects
        if self.max_obj_len < len(obj_pcds):
            # select target first
            if tgt_object_id != -1:
                selected_obj_idxs = [tgt_object_id]
            selected_obj_idxs.extend(tgt_object_id_iou25_list)
            selected_obj_idxs.extend(tgt_object_id_iou50_list)
            selected_obj_idxs = list(set(selected_obj_idxs))
            # select object with same semantic class with tgt_object
            remained_obj_idx = []
            for kobj, klabel in enumerate(obj_labels):
                if kobj not in selected_obj_idxs:
                    if klabel == tgt_object_label:
                        selected_obj_idxs.append(kobj)
                    else:
                        remained_obj_idx.append(kobj)
                if len(selected_obj_idxs) == self.max_obj_len:
                    break
            if len(selected_obj_idxs) < self.max_obj_len:
                random.shuffle(remained_obj_idx)
                selected_obj_idxs += remained_obj_idx[:(self.max_obj_len - len(selected_obj_idxs))]
            # reorganize ids
            obj_pcds = [obj_pcds[i] for i in selected_obj_idxs]
            obj_labels = [obj_labels[i] for i in selected_obj_idxs]
            if tgt_object_id != -1:
                tgt_object_id = selected_obj_idxs.index(tgt_object_id)
            tgt_object_id_iou25_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou25_list]
            tgt_object_id_iou50_list = [selected_obj_idxs.index(id)
                                        for id in tgt_object_id_iou50_list]
            assert len(obj_pcds) == self.max_obj_len

        # rebuild tgt_object_id
        if tgt_object_id == -1:
            tgt_object_id = len(obj_pcds)

        obj_fts, obj_locs, obj_boxes, obj_labels = self._obj_processing_post(obj_pcds, obj_labels, is_need_bbox=True)

        # build iou25 and iou50
        tgt_object_id_iou25 = torch.zeros(len(obj_fts) + 1).long()
        tgt_object_id_iou50 = torch.zeros(len(obj_fts) + 1).long()
        for _id in tgt_object_id_iou25_list:
            tgt_object_id_iou25[_id] = 1
        for _id in tgt_object_id_iou50_list:
            tgt_object_id_iou50[_id] = 1

        # build unique multiple
        is_multiple = self.scan_data[scan_id]['label_count'][tgt_object_label] > 1
        is_hard = self.scan_data[scan_id]['label_count'][tgt_object_label] > 2

        data_dict = {
            "sentence": sentence,
            "tgt_object_id": torch.LongTensor([tgt_object_id]), # 1
            "tgt_object_label": torch.LongTensor([tgt_object_label]), # 1
            "obj_fts": obj_fts, # N, 6
            "obj_locs": obj_locs, # N, 3
            "obj_labels": obj_labels, # N,
            "obj_boxes": obj_boxes, # N, 6 
            "data_idx": item_id,
            "tgt_object_id_iou25": tgt_object_id_iou25,
            "tgt_object_id_iou50": tgt_object_id_iou50, 
            'is_multiple': is_multiple,
            'is_view_dependent': is_view_dependent,
            'is_hard': is_hard
        }

        return data_dict

    def _load_lang(self, cfg, sources, split_scan_ids=None):
        lang_data = []
        scan_ids = set()

        if sources:
            if 'referit3d' in sources:
                for anno_type in cfg.referit3d.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/{anno_type}.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                                scan_ids.add(item['scan_id'])
                                lang_data.append(item)

                if cfg.referit3d.sr3d_plus_aug:
                    anno_file = os.path.join(self.base_dir, 'annotations/refer/sr3d+.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in split_scan_ids and len(item['tokens']) <= 24:
                                scan_ids.add(item['scan_id'])
                                lang_data.append(item)
            if 'scanrefer' in sources:
                anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
                with jsonlines.open(anno_file, 'r') as _f:
                    for item in _f:
                        if item['scan_id'] in split_scan_ids:
                            scan_ids.add(item['scan_id'])
                            lang_data.append(item)
            if 'sgrefer' in sources:
                for anno_type in cfg.sgrefer.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_{anno_type}_rels.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in split_scan_ids and item['instance_type'] not in ['wall', 'floor', 'ceiling']:
                            scan_ids.add(item['scan_id'])
                            lang_data.append(item)
            if 'sgcaption' in sources:
                for anno_type in cfg.sgcaption.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_{anno_type}_caption_sumi.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in split_scan_ids and item['instance_type'] not in ['wall', 'floor', 'ceiling']:
                            scan_ids.add(item['scan_id'])
                            lang_data.append(item)

        return lang_data, scan_ids


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
class ScanNetSGQA(ScanNetBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self.dataset_cfg = cfg.data.scannet_sgqa.args
        
        self.num_points = self.dataset_cfg.get('num_points', 1024)
        self.max_obj_len = self.dataset_cfg.get('max_obj_len', 60)
        self.val_num = self.dataset_cfg.get('val_num', 100)
        self.pc_type = self.dataset_cfg.get('pc_type', 'gt')

        assert self.pc_type in ['gt', 'pred']
        assert self.split in ['train', 'val', 'test']
        if self.split == 'train':
            self.pc_type = 'gt'
        
        print(f"Loading ScanNet SGQA3D {split}-set language")
        self.data, self.scan_ids = self._load_lang(self.dataset_cfg.anno_dir, split)
        if cfg.debug.flag:
            self.data = self.data[:cfg.debug.debug_size]
        print(f"Finish loading ScanNet SGQA {split}-set language")
        
        # load scans
        print(f"Loading ScanNet SGQA {split}-set scans")
        self.scan_data = self._load_scannet(self.scan_ids, self.pc_type, self.pc_type == 'gt')
        print(f"Finish loading ScanNet SGQA {split}-set data")
    
    def _load_lang(self, anno_dir, split):
        output_list = []
        fname = f'sgqa_{split}.json'
        scan_ids = []
        with open(os.path.join(anno_dir, fname)) as f:
            json_data = json.load(f)
        for k, v in json_data.items():
            if 'response' not in v:
                continue
            for meta_anno in v['response']:
                # try to parse concerned objects
                try:
                    insts = meta_anno['T'].split(', ')
                    insts = [int(s.split('-')[-1]) for s in insts]
                except:
                    insts = []
                
                meta_anno['A'] = [a.strip() for a in meta_anno['A']]
                output_list.append({
                    'scan_id' : k, 
                    'qa_pair': meta_anno, 
                    'insts': insts
                })
            scan_ids.append(k)
        scan_ids = list(set(scan_ids))                      
        return output_list, scan_ids
    
    def __len__(self):
        return len(self.data)

    # get inputs for scene encoder
    def preprocess_pcd(self, obj_pcds, return_anchor=False, rot_aug=True):
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

        return obj_fts, obj_locs, anchor_loc

    def _get_scene_encoder_input(self, obj_pcds, scan_insts):

        # Dict: { int: np.ndarray (N, 6) }
        if len(obj_pcds) <= self.max_obj_len:
            # Dict to List
            selected_obj_pcds = list(obj_pcds.values())
        else:
            # crop objects to max_obj_len
            selected_obj_pcds = []

            # select relevant objs first
            for i in scan_insts:
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

        obj_fts, obj_locs, anchor_loc = self.preprocess_pcd(selected_obj_pcds, return_anchor=False)
        return obj_fts, obj_locs, anchor_loc

    def __getitem__(self, index):
        one_sample = self.data[index]
        question = one_sample['qa_pair']['Q']
        answer_list = one_sample['qa_pair']['A']
        situation = one_sample['qa_pair']['situation']
        anchor_loc = one_sample['qa_pair']['location']
        anchor_orientation = one_sample['qa_pair']['orientation'] ### face pt
        anchor_orientation = face_vector_in_xy_to_quaternion(anchor_orientation)
        question_type = get_sqa_question_type(question)

        scan_id = one_sample['scan_id']

        ### load data with global cache ###
        scan_data = self.scan_data[scan_id]
        obj_pcds = scan_data['obj_pcds']
        obj_pcds = {i: obj_pcds[i] for i in range(len(obj_pcds))}

        ### scene input ####
        obj_fts, obj_locs, _ = self._get_scene_encoder_input(obj_pcds, one_sample['insts'])
        
        data_dict = {
            "situation": situation,
            "situation_pos": np.array(anchor_loc),
            "situation_rot": anchor_orientation,
            "question": question,
            "answer_list": "[answer_seq]".join(answer_list),
            "obj_fts": obj_fts,
            "obj_locs": obj_locs,
            "sqa_type": question_type,
        }

        return data_dict