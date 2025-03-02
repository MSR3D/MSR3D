import os
import torch
from PIL import Image
import cv2
import numpy as np
from ..data_utils import build_rotate_mat, preprocess_2d
from .scannet_base import ScanNetBase
# from torchvision.transforms import v2
from transformers import CLIPProcessor
import cv2
import json

PIX_MEAN = (0.485, 0.456, 0.406)
PIX_STD = (0.229, 0.224, 0.225)
### class for loading data from different datasets ###
### support dataset : [ScanNet] ###
class ScanDataLoader(object):
    def __init__(self, cfg, dataset = '') -> None:
        if dataset == 'ScanNet':
            self.scannet_loader = ScanNetBase(cfg, split = 'train')
        elif dataset in ['3RScan', 'ARkit']:
            self.cfg = cfg
        else:
            raise NotImplementedError(f"loading dataset {dataset} not supported")
        
        self.cfg = cfg
        self.img_process_args = cfg.data.process_args.img_process_args
        self.bbox_keep_ratio = self.img_process_args.get('bbox_keep_ratio', 0.5)
        self.min_keep_num = self.img_process_args.get('min_keep_num', 1)
        self.bbox_expand = self.img_process_args.get('bbox_expand', 0.1)
        self.img_processer_type = self.img_process_args.get("img_processer", "openai/clip-vit-base-patch32")
        self.dataset = dataset

        if self.img_processer_type in ['openai/clip-vit-base-patch32']:
            self.img_transform = CLIPProcessor.from_pretrained(self.img_processer_type).image_processor
        elif self.img_processer_type in ['navigation_img_processer']:
            pass
        else:
            raise NotImplementedError
            
    def get_data(self, dataset, scan_id, data_type = ['obj_pcds', 'mv_info'], pc_type = 'gt'):
        if dataset == 'ScanNet':
            return self._get_scannet_data(scan_id, pc_type = pc_type, data_type = data_type)
        elif dataset == '3RScan':
            return self._get_rscan_data(scan_id, pc_type = pc_type, data_type = data_type)
        elif dataset == 'ARkit':
            return self._get_arkit_data(scan_id, pc_type = pc_type, data_type = data_type)
        else:
            raise NotImplementedError(f"{dataset} not supported")
    
    def _get_rscan_data(self, scan_id, pc_type = 'gt', data_type = ['obj_pcds', 'mv_info']):
        scan_data = {}
        if 'mv_info' in data_type:
            ### load mv info
            mv_info_path = os.path.join(self.cfg.data.mv_info_base, "3RScan_caption_with_object", scan_id, "cap_res.json")
            with open(mv_info_path, 'r') as f:
                mv_info_all = json.load(f)
            
            obj_dict = {}
            for inst_id in mv_info_all.keys():
                for one_bbox in mv_info_all[inst_id]:
                    bbox_2d = one_bbox["bbox"]
                    frame_path = one_bbox["frame_path"]
                    frame_name = one_bbox["frame"]
                    one_bbox_to_save = {
                        'bbox_2d': bbox_2d,
                        'inst_id': inst_id,
                        'frame_name': frame_name,
                        'frame_path': frame_path,
                        'label': one_bbox["tgt_label"],
                    }
                    if int(inst_id) not in obj_dict.keys():
                        obj_dict[int(inst_id)] = []
                    obj_dict[int(inst_id)].append(one_bbox_to_save)

            #### sort the inst proj list by size of bbox from large to small for better ####
            for inst_id in obj_dict.keys():
                obj_dict[inst_id] = sorted(obj_dict[inst_id], key = lambda x: (x['bbox_2d'][1][0] - x['bbox_2d'][0][0]) * (x['bbox_2d'][1][1] - x['bbox_2d'][0][1]), reverse = True)
                obj_dict[inst_id] = obj_dict[inst_id][: max(self.min_keep_num, int(len(obj_dict[inst_id]) * self.bbox_keep_ratio)) + 1]
                
            scan_data['mv_info'] = obj_dict

        if 'obj_pcds' in data_type:
            pcd_data = torch.load(os.path.join(self.cfg.data.rscan_base, "3RScan-ours-align", scan_id, "pcds.pth"))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # build obj_pcds
            inst_to_label = torch.load(os.path.join(self.cfg.data.rscan_base, "3RScan-ours-align", scan_id,"inst_to_label.pth"))
            obj_pcds = {}
            for inst_id in inst_to_label.keys():
                mask = instance_labels == inst_id
                obj_pcds.update({inst_id: pcds[mask]})
            scan_data['obj_pcds'] = obj_pcds
        return scan_data

    def _get_arkit_data(self, scan_id, pc_type = 'gt', data_type = ['obj_pcds', 'mv_info']):
        scan_data = {}
        if 'mv_info' in data_type:
            ### load mv info
            mv_info_path = os.path.join(self.cfg.data.mv_info_base, "ARkit_caption_for_EQA", "arkit_unique", scan_id, "frame_bbox.json")
            mv_img_dir = os.path.join(self.cfg.data.mv_info_base, "ARkit_caption_for_EQA", "arkit_unique", scan_id, "vga_wide", "vga_wide")
            with open(mv_info_path, 'r') as f:
                mv_info_all = json.load(f)
            
            mv_info_all = self.transfer_frame_to_obj(mv_info_all)

            obj_dict = {}
            for inst_id in mv_info_all.keys():
                for one_bbox in mv_info_all[inst_id]:
                    bbox_2d = one_bbox["bbox"]
                    frame_id = one_bbox["frame_id"]
                    frame_name = f"{scan_id}_{frame_id}.png"
                    frame_name = frame_name
                    frame_path = os.path.join(mv_img_dir, frame_name)
                    one_bbox_to_save = {
                        'bbox_2d': bbox_2d,
                        'inst_id': inst_id,
                        'frame_name': frame_name,
                        'frame_path': frame_path,
                        'label': one_bbox["cls_label"],
                    }
                    if int(inst_id) not in obj_dict.keys():
                        obj_dict[int(inst_id)] = []
                    obj_dict[int(inst_id)].append(one_bbox_to_save)

            #### sort the inst proj list by size of bbox from large to small for better ####
            for inst_id in obj_dict.keys():
                obj_dict[inst_id] = sorted(obj_dict[inst_id], key = lambda x: (x['bbox_2d'][1][0] - x['bbox_2d'][0][0]) * (x['bbox_2d'][1][1] - x['bbox_2d'][0][1]), reverse = True)
                obj_dict[inst_id] = obj_dict[inst_id][: max(self.min_keep_num, int(len(obj_dict[inst_id]) * self.bbox_keep_ratio)) + 1]
                
            scan_data['mv_info'] = obj_dict

        if 'obj_pcds' in data_type:
            pcd_data = torch.load(os.path.join(self.cfg.data.ARkit_base, "scan_data", "pcd-align", f"{scan_id}.pth"))
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[2]
            colors = colors / 127.5 - 1
            pcds = np.concatenate([points, colors], 1)
            # build obj_pcds
            inst_to_label = torch.load(os.path.join(self.cfg.data.ARkit_base, "scan_data", "instance_id_to_label", f"{scan_id}_inst_to_label.pth"))
            obj_pcds = {}
            for inst_id in inst_to_label.keys():
                if type(inst_id) != int:
                    continue
                mask = instance_labels == inst_id
                if mask.sum() < 10:
                    continue
                obj_pcds.update({inst_id: pcds[mask]})
            scan_data['obj_pcds'] = obj_pcds

        return scan_data

    def _get_scannet_data(self, scan_id, pc_type = 'gt', data_type = ['obj_pcds', 'mv_info']):
        _, scan_data = self.scannet_loader._load_one_scan(scan_id, 
                        pc_type = pc_type, load_inst_info = True, 
                        load_multiview_info = ('mv_info' in data_type), is_load_mv_feat = False, 
                        load_pc_info = ('obj_pcds' in data_type), 
                        load_segment_info = False)
        
        ### transfer data to inst for mv_info ###
        if 'mv_info' in data_type:
            mv_info_all = scan_data.pop('multiview_info')
            obj_dict = {}
            for frame_name, frame_info in mv_info_all.items():
                inst_all = [x for x in frame_info['instance_info'] if x['is_need_process']]
                for one_inst in inst_all:
                    bbox_2d = one_inst['bbox']
                    tmp_inst_id = one_inst['org_inst_id']
                    inst_id = int(self.scannet_loader.label_converter.orgInstID_to_id[tmp_inst_id])
                    if os.path.exists(os.path.join(self.scannet_loader.base_dir, 'ScanNetV2-RGBD/scans_RGBD_deblur', scan_id, str(frame_name) + '.jpg')):
                        frame_path = os.path.join(self.scannet_loader.base_dir, 'ScanNetV2-RGBD/scans_RGBD_deblur', scan_id, str(frame_name) + '.jpg')
                    elif os.path.exists(os.path.join(self.scannet_loader.base_dir, 'ScanNetV2-RGBD/org_frame_data', scan_id, str(frame_name) + '.jpg')):
                        frame_path = os.path.join(self.scannet_loader.base_dir, 'ScanNetV2-RGBD/org_frame_data', scan_id, str(frame_name) + '.jpg')
                    one_bbox = {
                        'bbox_2d': bbox_2d,
                        'inst_id': inst_id,
                        'frame_name': frame_name,
                        'frame_path': frame_path,
                    }
                    if inst_id not in obj_dict.keys():
                        obj_dict[inst_id] = []
                    obj_dict[inst_id].append(one_bbox)

            #### sort the inst proj list by size of bbox from large to small for better ####
            for inst_id in obj_dict.keys():
                obj_dict[inst_id] = sorted(obj_dict[inst_id], key = lambda x: (x['bbox_2d'][1][0] - x['bbox_2d'][0][0]) * (x['bbox_2d'][1][1] - x['bbox_2d'][0][1]), reverse = True)
                obj_dict[inst_id] = obj_dict[inst_id][:int(len(obj_dict[inst_id]) * self.bbox_keep_ratio)]
                
            scan_data['mv_info'] = obj_dict

        if 'obj_pcds' in data_type:
            scan_data['obj_pcds'] = {idx: scan_data['obj_pcds'][idx] for idx in range(len(scan_data['obj_pcds']))}

        return scan_data

    def preprocess_2d(img, size=(224, 224)):
        # lhxk: this function is copied from data_utils of LEO, to align with the image encoder in LEO
        # img: (H, W, 3)
        # resize, normalize, to pytorch tensor format

        img = cv2.resize(img, size).astype(np.float32)
        for i in range(3):
            img[:, :, i] = (img[:, :, i] / 255.0 - PIX_MEAN[i]) / PIX_STD[i]
        return np.ascontiguousarray(img.transpose(2, 0, 1))

    def get_one_img(self, one_bbox, is_debug = False):
        img = Image.open(one_bbox['frame_path'])
        img_w, img_h = img.size
        [l, t], [r, b] = one_bbox['bbox_2d']
        bbox_w, bbox_h = r - l, b - t
        l -= bbox_w * self.bbox_expand
        r += bbox_w * self.bbox_expand
        t -= bbox_h * self.bbox_expand
        b += bbox_h * self.bbox_expand
        l, t, r, b = int(max(0, l)), int(max(0, t)), int(min(img_w-1, r)), int(min(img_h-1, b))
        img = img.crop((l, t, r, b))

        if self.img_processer_type in ['openai/clip-vit-base-patch32']:
            img = self.img_transform(img)
            img = img['pixel_values'][0]
        elif self.img_processer_type in ['navigation_img_processer']:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            ### debug ###
            if is_debug:
                cv2.imwrite(f'img_{one_bbox["label"]}_img.png', img)

            img = preprocess_2d(img, size=self.img_process_args.tgt_img_size)
        else:
            raise NotImplementedError

        img = torch.from_numpy(img)
        return img

    def get_one_certain_img(self, scan_id, inst_id, label, is_debug = False):
        img_file_name = f'{scan_id}_inst{inst_id}_{label}_0.jpg'
        img_path = os.path.join(self.cfg.data.obj_img_base, self.dataset, img_file_name)
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path)

        if self.img_processer_type in ['openai/clip-vit-base-patch32']:
            img = self.img_transform(img)
            img = img['pixel_values'][0]
        elif self.img_processer_type in ['navigation_img_processer']:
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = preprocess_2d(img, size=self.img_process_args.tgt_img_size)
        else:
            raise NotImplementedError

        img = torch.from_numpy(img)
        return img

    def get_one_pcd(self, obj_pcd, rot_aug = True, num_points = 1024):
        rot_matrix = build_rotate_mat(split = 'train', rot_aug = rot_aug)

        if rot_matrix is not None:
            obj_pcd[:, :3] = np.matmul(obj_pcd[:, :3], rot_matrix.transpose())

        obj_center = obj_pcd[:, :3].mean(0)
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        obj_loc = np.concatenate([obj_center, obj_size], 0)

        # subsample
        pcd_idxs = np.random.choice(len(obj_pcd), size = num_points,
                                    replace=len(obj_pcd) < num_points)
        obj_pcd = obj_pcd[pcd_idxs]

        # normalize
        obj_pcd[:, :3] = obj_pcd[:, :3] - obj_pcd[:, :3].mean(0)
        max_dist = np.sqrt((obj_pcd[:, :3]**2).sum(1)).max()
        if max_dist < 1e-6:   # take care of tiny point-clouds, i.e., padding
            max_dist = 1
        obj_pcd[:, :3] = obj_pcd[:, :3] / max_dist

        obj_pcd = torch.from_numpy(obj_pcd).float()
        obj_loc = torch.from_numpy(obj_loc).float()
        
        return obj_pcd, obj_loc

    @staticmethod
    def transfer_frame_to_obj(frame_dict):
        inst_dict = {}
        for frame_id, bbox_info_list in frame_dict.items():
            for one_inst in bbox_info_list:
                inst_id = one_inst['inst_id']
                if inst_id in inst_dict:
                    inst_dict[inst_id].append(one_inst)
                else:
                    inst_dict[inst_id] = [one_inst]
        return inst_dict