import os
import collections
import json
import pickle
import random

import jsonlines
from tqdm import tqdm
from scipy import sparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from common.misc import rgetattr
from ..data_utils import (convert_pc_to_box, LabelConverter, build_rotate_mat, load_matrix_from_txt)
import einops 

class ScanNetBase(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.base_dir = cfg.data.scan_family_base
        assert self.split in ['train', 'val', 'test']

        self.int2cat = json.load(open(os.path.join(self.base_dir,
                                            "annotations/meta_data/scannetv2_raw_categories.json"),
                                            'r', encoding="utf-8"))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        self.label_converter = LabelConverter(os.path.join(self.base_dir,
                                            "annotations/meta_data/scannetv2-labels.combined.tsv"))

        # self.referit3d_camera_pose = json.load(open(os.path.join(self.base_dir,
        #                                     "annotations/meta_data/scans_axis_alignment_matrices.json"),
        #                                     'r', encoding="utf-8"))
        self.rot_matrix = build_rotate_mat(self.split)
        self.use_cache = rgetattr(self.cfg.data, 'mvdatasettings.use_cache', False)
        self.cache = {}

    def __len__(self):
        return len(self.lang_data)

    def __getitem__(self, index):
        raise NotImplementedError

    def _load_one_scan(self, scan_id, pc_type = 'gt', load_inst_info = False, 
                      load_multiview_info = False, is_load_mv_feat = True, load_pc_info = True, load_segment_info=False):
        one_scan = {}
        if load_inst_info:
            inst_labels, inst_locs, inst_colors = self._load_inst_info(scan_id)
            one_scan['inst_labels'] = inst_labels # (n_obj, )
            one_scan['inst_locs'] = inst_locs # (n_obj, 6) center xyz, whl
            one_scan['inst_colors'] = inst_colors # (n_obj, 3x4) cluster * (weight, mean rgb)

        if load_pc_info:
            # load pcd data
            pcd_data = torch.load(os.path.join(self.base_dir, "scan_data",
                                            "pcd_with_global_alignment", f'{scan_id}.pth'))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # convert to gt object
            if load_inst_info:
                obj_pcds = []
                for i in range(instance_labels.max() + 1):
                    mask = instance_labels == i     # time consuming
                    obj_pcds.append(pcds[mask])
                one_scan['obj_pcds'] = obj_pcds
                # calculate box for matching
                obj_center = []
                obj_box_size = []
                for obj_pcd in obj_pcds:
                    _c, _b = convert_pc_to_box(obj_pcd)
                    obj_center.append(_c)
                    obj_box_size.append(_b)
                one_scan['obj_center'] = obj_center
                one_scan['obj_box_size'] = obj_box_size
            if pc_type == 'pred':
                obj_mask_path = os.path.join(self.base_dir, "mask", str(scan_id) + ".mask" + ".npz")
                obj_label_path = os.path.join(self.base_dir, "mask",
                                            str(scan_id) + ".label" + ".npy")
                obj_pcds = []
                obj_mask = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
                obj_labels = np.load(obj_label_path)[:50]
                obj_l = []
                for i in range(obj_mask.shape[0]):
                    mask = obj_mask[i]
                    if pcds[mask == 1, :].shape[0] > 0:
                        obj_pcds.append(pcds[mask == 1, :])
                        obj_l.append(obj_labels[i])
                one_scan['obj_pcds_pred'] = obj_pcds
                one_scan['inst_labels_pred'] = obj_l
                # calculate box for pred
                obj_center_pred = []
                obj_box_size_pred = []
                for obj_pcd in obj_pcds:
                    _c, _b = convert_pc_to_box(obj_pcd)
                    obj_center_pred.append(_c)
                    obj_box_size_pred.append(_b)
                one_scan['obj_center_pred'] = obj_center_pred
                one_scan['obj_box_size_pred'] = obj_box_size_pred
                ##################
                # obj_pcds = []
                # obj_mask = np.load(obj_mask_path)
                # obj_labels = np.load(obj_label_path)
                # obj_labels = [self.label_converter.nyu40id_to_id[int(l)] for l in obj_labels]
                # for i in range(obj_mask.shape[0]):
                #     mask = obj_mask[i]
                #     obj_pcds.append(pcds[mask == 1, :])
                # one_scan['obj_pcds_pred'] = obj_pcds
                # one_scan['inst_labels_pred'] = obj_labels
                # # calculate box for pred
                # obj_center_pred = []
                # obj_box_size_pred = []
                # for obj_pcd in obj_pcds:
                #     _c, _b = convert_pc_to_box(obj_pcd)
                #     obj_center_pred.append(_c)
                #     obj_box_size_pred.append(_b)
                # one_scan['obj_center_pred'] = obj_center_pred
                # one_scan['obj_box_size_pred'] = obj_box_size_pred
                ##################

        if load_multiview_info:
            one_scan['multiview_info'] = self._load_multiview_info(scan_id, is_load_mv_feat = is_load_mv_feat)
        
        # load segment for mask3d
        if load_segment_info:
            one_scan["scene_pcds"] = np.load(os.path.join(self.base_dir, "scan_data", "pcd_mask3d", f'{scan_id[-7:]}.npy'))
        ######### testing ########

        return (scan_id, one_scan)

    def _load_scannet(self, scan_ids, pc_type = 'gt', load_inst_info = False, 
                      load_multiview_info = False, load_pc_info = True, load_segment_info = False, 
                      use_multi_process = False, process_num = 20):
        scans = {}
        if use_multi_process:
            from joblib import Parallel, delayed
            res_all = Parallel(n_jobs=process_num)(
                delayed(self._load_one_scan)(scan_id, pc_type = pc_type,
                                        load_inst_info = load_inst_info,
                                        load_multiview_info = load_multiview_info,
                                        load_pc_info = load_pc_info, load_segment_info=load_segment_info) for scan_id in tqdm(scan_ids))
            for scan_id, one_scan in tqdm(res_all):
                scans[scan_id] = one_scan

        else:
            for scan_id in tqdm(scan_ids):
                _, one_scan = self._load_one_scan(scan_id, pc_type = pc_type,
                                                  load_inst_info = load_inst_info, 
                                                  load_multiview_info = load_multiview_info, 
                                                  load_pc_info = load_pc_info, load_segment_info=load_segment_info)
                scans[scan_id] = one_scan

        return scans

    def _load_lang(self, cfg):
        caption_source = cfg.sources
        lang_data = []
        if caption_source:
            if 'scanrefer' in caption_source:
                anno_file = os.path.join(self.base_dir, 'annotations/refer/scanrefer.jsonl')
                with jsonlines.open(anno_file, 'r') as _f:
                    for item in _f:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'referit3d' in caption_source:
                for anno_type in cfg.referit3d.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/{anno_type}.jsonl')
                    with jsonlines.open(anno_file, 'r') as _f:
                        for item in _f:
                            if item['scan_id'] in self.scannet_scan_ids:
                                lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'scanqa' in caption_source:
                anno_file_list = ['annotations/qa/ScanQA_v1.0_train.json',
                                  'annotations/qa/ScanQA_v1.0_val.json']
                for anno_file in anno_file_list:
                    anno_file = os.path.join(self.base_dir, anno_file)
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scene_id'] in self.scannet_scan_ids:
                            for i in range(len(item['answers'])):
                                lang_data.append(('scannet', item['scene_id'],
                                                  item['question'] + " " + item['answers'][i]))

            if 'sgrefer' in caption_source:
                for anno_type in cfg.sgrefer.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_ref_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))

            if 'sgcaption' in caption_source:
                for anno_type in cfg.sgcaption.anno_type:
                    anno_file = os.path.join(self.base_dir,
                                             f'annotations/refer/ssg_caption_{anno_type}.json')
                    json_data = json.load(open(anno_file, 'r', encoding='utf-8'))
                    for item in json_data:
                        if item['scan_id'] in self.scannet_scan_ids:
                            lang_data.append(('scannet', item['scan_id'], item['utterance']))
        return lang_data

    def _load_multiview_info(self, scan_id, is_load_mv_feat = True):
        # multiview_info_dir = os.path.join(self.base_dir, 'ScanNetV2-RGBD/org_frame_data', scan_id, 'instance_feature')
        # camera_pose_dir = os.path.join(self.base_dir, 'ScanNetV2-RGBD/org_frame_data', scan_id, scan_id + '_pose')
        # multiview_info = {}
        # if os.path.exists(multiview_info_dir):
        #     frame_list = os.listdir(multiview_info_dir)
        #     multiview_info = {}
        #     for one_frame in frame_list:
        #         frame_name = one_frame.split('.')[0]
        #         frame_json_path = os.path.join(multiview_info_dir, one_frame)
        #         with open(frame_json_path, 'r') as f:
        #             frame_info = json.load(f)
        #         ### load camera pose
        #         camera_to_world = load_matrix_from_txt(os.path.join(camera_pose_dir, frame_name + '.txt'))
        #         referit3d_matrix = np.array(self.referit3d_camera_pose[scan_id], dtype=np.float32).reshape(4, 4)
        #         camera_pose_mat = np.matmul(np.linalg.inv(camera_to_world), np.linalg.inv(referit3d_matrix))
        #         frame_info['camera_pose'] = camera_pose_mat.flatten()

        #         multiview_info[frame_name] = frame_info

        ### load one scan file
        # json_path = os.path.join(self.base_dir, 'ScanNetV2-RGBD/MultiViewInfo', scan_id + '.json')
        # with open(json_path, 'r') as f:
        #     multiview_info = json.load(f)

        ############### load numpy hashing file ###############
        json_path = os.path.join(self.base_dir, 'ScanNetV2-RGBD/MultiViewInfo_numpy', scan_id, 'multiview_info_refined.json')
        with open(json_path, 'r') as f:
            scan_info = json.load(f)
        # pkl_path = os.path.join(self.base_dir, 'ScanNetV2-RGBD/MultiViewInfo_numpy', scan_id, 'multiview_info.json')
        # with open(pkl_path, 'rb') as f:
        #     scan_info = pickle.load(f)
        multiview_info = scan_info['multiview_info']

        if is_load_mv_feat:
            feature_np_path = os.path.join(self.base_dir, 'ScanNetV2-RGBD/MultiViewInfo_numpy', scan_id, f"{self.cfg.data.mvdatasettings.inst_feat_type}"".npy")
            feature_np = np.load(feature_np_path)

            for frame_name in multiview_info.keys():
                frame_name = str(frame_name)
                for inst_idx in range(len(multiview_info[frame_name]['instance_info'])):
                    if not multiview_info[frame_name]['instance_info'][inst_idx]['is_need_process']:
                        continue
                    feat_hashing_idx = multiview_info[frame_name]['instance_info'][inst_idx][self.cfg.data.mvdatasettings.inst_feat_type]
                    multiview_info[frame_name]['instance_info'][inst_idx][self.cfg.data.mvdatasettings.inst_feat_type] = feature_np[feat_hashing_idx]

        return multiview_info

    def _load_split(self, cfg, split, use_multi_process = False):
        if use_multi_process and split in ['train']:
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + "_sort.json")
            with open(split_file, 'r') as f:
                scannet_scan_ids = json.load(f)
        else:
            split_file = os.path.join(self.base_dir, 'annotations/splits/scannetv2_'+ split + ".txt")
            scannet_scan_ids = {x.strip() for x in open(split_file, 'r', encoding="utf-8")}
            scannet_scan_ids = sorted(scannet_scan_ids)

        if cfg.debug.flag and cfg.debug.debug_size != -1:
            scannet_scan_ids = list(scannet_scan_ids)[:cfg.debug.debug_size]
        return scannet_scan_ids

    def _load_inst_info(self, scan_id):
        inst_labels = json.load(open(os.path.join(self.base_dir, 'scan_data',
                                                    'instance_id_to_name',
                                                    f'{scan_id}.json'), encoding="utf-8"))
        inst_labels = [self.cat2int[i] for i in inst_labels]

        inst_locs = np.load(os.path.join(self.base_dir, 'scan_data',
                                            'instance_id_to_loc', f'{scan_id}.npy'))
        inst_colors = json.load(open(os.path.join(self.base_dir, 'scan_data',
                                                    'instance_id_to_gmm_color',
                                                    f'{scan_id}.json'), encoding="utf-8"))
        inst_colors = [np.concatenate(
            [np.array(x['weights'])[:, None], np.array(x['means'])],
            axis=1).astype(np.float32) for x in inst_colors]

        return inst_labels, inst_locs, inst_colors

    def _obj_processing_post(self, obj_pcds, obj_labels, is_need_bbox=False, rot_aug=True, situation=None):
        # rotate obj
        rot_matrix = build_rotate_mat(self.split, rot_aug)

        # normalize pc and calculate location
        obj_fts = []
        obj_locs = []
        obj_boxes = []
        for obj_pcd in obj_pcds:
            # build locs
            if rot_matrix is not None:
                obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())
            obj_center = obj_pcd[:, :3].mean(0)
            obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
            obj_locs.append(np.concatenate([obj_center, obj_size], 0))

            # build box
            if is_need_bbox:
                obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                obj_boxes.append(np.concatenate([obj_box_center, obj_box_size], 0))

            # subsample
            pcd_idxs = np.random.choice(len(obj_pcd), size=self.num_points,
                                        replace=len(obj_pcd) < self.num_points)
            obj_pcd = obj_pcd[pcd_idxs]
            # normalize
            obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
            max_dist = np.max(np.sqrt(np.sum(obj_pcd[:, :3]**2, 1)))
            if max_dist < 1e-6: # take care of tiny point-clouds, i.e., padding
                max_dist = 1
            obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist
            obj_fts.append(obj_pcd)

        # convert to torch
        obj_fts = torch.from_numpy(np.stack(obj_fts, 0))
        obj_locs = torch.from_numpy(np.array(obj_locs))
        obj_boxes = torch.from_numpy(np.array(obj_boxes))
        obj_labels = torch.LongTensor(obj_labels)

        assert obj_labels.shape[0] == obj_locs.shape[0]
        assert obj_fts.shape[0] == obj_locs.shape[0]

        if situation is None:
            return obj_fts, obj_locs, obj_boxes, obj_labels
        elif rot_matrix is None:
            return obj_fts, obj_locs, obj_boxes, obj_labels, situation
        else:
            pos, ori = situation
            pos_new = pos.reshape(1, 3) @ rot_matrix.transpose()
            pos_new = pos_new.reshape(-1)
            ori_new = R.from_quat(ori).as_matrix()
            ori_new = rot_matrix @ ori_new
            ori_new = R.from_matrix(ori_new).as_quat()
            ori_new = ori_new.reshape(-1)
            return obj_fts, obj_locs, obj_boxes, obj_labels, (pos_new, ori_new)

    def _get_multiview_info(self, scan_id):
        if self.use_cache and scan_id in self.cache.keys():
            return self.cache[scan_id]
        
        mv_info_all = self.scan_data[scan_id]['multiview_info']
        frame_names = mv_info_all.keys()
        args = self.cfg.data.mvdatasettings

        if args.frame_sample_mode == 'even':
            max_frame_num = min(args.max_frame_num, len(frame_names))
            sampled_frame_names = random.sample(frame_names, max_frame_num)
        else:
            raise ValueError

        #### fusing instance features to obj
        if args.is_pool_obj_feature:
            out_dict = self._get_pooling_obj_feature(args, mv_info_all, sampled_frame_names, scan_id)
        else:
            out_dict = self._get_inst_features(args, mv_info_all, sampled_frame_names, scan_id)

        if self.use_cache:
            self.cache[scan_id] = out_dict

        return out_dict 

    def _get_pooling_obj_feature(self, args, mv_info_all, sampled_frame_names, scan_id):
        obj_dict = {}
        for i in range(len(sampled_frame_names)):
            frame_info = mv_info_all[sampled_frame_names[i]]
            inst_all = [x for x in frame_info['instance_info'] if x['is_need_process']]
            for one_inst in inst_all:
                tmp_inst_id = one_inst['org_inst_id']
                feat = one_inst[args.inst_feat_type]
                feat = feat[0] if len(feat) == 1 else feat

                inst_id = self.label_converter.orgInstID_to_id[tmp_inst_id]
                if inst_id in obj_dict.keys():
                    obj_dict[inst_id]['feat'].append(feat)
                    assert self.scan_data[scan_id]['inst_labels'][inst_id] == obj_dict[inst_id]['label']
                else:
                    obj_pcd = self.scan_data[scan_id]['obj_pcds'][inst_id]
                    if self.rot_matrix is not None:
                        obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], self.rot_matrix.transpose())
                    obj_center = obj_pcd[:, :3].mean(0)
                    obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                    obj_loc = np.concatenate([obj_center, obj_size], 0)

                    obj_box_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
                    obj_box_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
                    obj_box = np.concatenate([obj_box_center, obj_box_size], 0)

                    obj_dict[inst_id] = {
                        'feat': [feat],
                        'location': obj_loc,
                        'label': self.scan_data[scan_id]['inst_labels'][inst_id],
                        'box' : obj_box,
                    }

        if args.pooling_strategy == 'average_all':
            for key in obj_dict.keys():
                feat_all = np.array(obj_dict[key]['feat'])
                if args.pooling_strategy == 'average_all':
                    obj_dict[key]['feat'] = np.mean(feat_all, axis = 0)

        # for key in obj_dict.keys():
        #     obj_dict[key]['feat'] = list(obj_dict[key]['feat'])
        #     obj_dict[key]['location'] = list(obj_dict[key]['location'])
        #     obj_dict[key]['box'] = list(obj_dict[key]['box'])

        return obj_dict

    def _get_inst_features(self, args, mv_info_all, sampled_frame_names, scan_id):
        out_feat = np.zeros((args.max_frame_num, args.max_inst_per_frame, args.inst_feat_len))
        out_mask = np.zeros((args.max_frame_num, args.max_inst_per_frame))
        out_inst_loc = np.zeros((args.max_frame_num, args.max_inst_per_frame, 3))
        out_camera_pose = np.zeros((args.max_frame_num, args.max_inst_per_frame, 16))
        out_cls = np.zeros((args.max_frame_num, args.max_inst_per_frame)) - 100
    
        for i in range(len(sampled_frame_names)):
            frame_info = mv_info_all[sampled_frame_names[i]]
            inst_all = [x for x in frame_info['instance_info'] if x['is_need_process']]
            if args.inst_sample_mode == 'even':
                max_inst_num = min(args.max_inst_per_frame, len(inst_all))
                sampled_inst_list = random.sample(inst_all, max_inst_num)
                for j in range(len(sampled_inst_list)):
                    feat = sampled_inst_list[j][args.inst_feat_type]
                    feat = feat[0] if len(feat) == 1 else feat
                    out_feat[i][j] = np.array(feat)
                    out_mask[i][j] = 1
                    if args.inst_position_type in ['pc_gt']:
                        tmp_inst_id = sampled_inst_list[j]['org_inst_id']
                        inst_id = self.label_converter.orgInstID_to_id[tmp_inst_id]
                        out_inst_loc[i][j] = self.scan_data[scan_id]['obj_center'][inst_id]
                        out_cls[i][j] = self.scan_data[scan_id]['inst_labels'][inst_id]
            out_camera_pose[i, :, :] = np.tile(frame_info['camera_pose'], (args.max_inst_per_frame, 1))
    
        out_feat = einops.rearrange(out_feat, 'f i l -> (f i) l')
        out_inst_loc = einops.rearrange(out_inst_loc, 'f i l -> (f i) l')
        out_camera_pose = einops.rearrange(out_camera_pose, 'f i l -> (f i) l')
        out_mask = einops.rearrange(out_mask, 'f i -> (f i)')
        out_cls = einops.rearrange(out_cls, 'f i -> (f i)')

        out_dict = {
            'mv_inst_feats' : out_feat, 
            'mv_inst_masks' : out_mask, 
            'mv_inst_locs' : out_inst_loc, 
            'mv_camera_pose' : out_camera_pose, 
            'mv_inst_labels' : out_cls,
        }

        return out_dict