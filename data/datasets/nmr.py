import os
import random
import yaml

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .default import DATASET_REGISTRY
from ..data_utils import transform_points


@DATASET_REGISTRY.register()
class NMRMVRecon(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.base_dir = cfg.data.nmr_base
        self.canonical = cfg.data.mvrecon.args.canonical
        self.full_scale = cfg.data.mvrecon.args.full_scale
        self.points_per_item = cfg.data.mvrecon.args.points_per_item
        # Rotation matrix making z=0 is the ground plane.
        # Ensures that the scenes are layed out in the same way as the other datasets,
        # which is convenient for visualization.
        self.rot_mat = np.array([[1, 0, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]])
        self.load_nmr()
        self.num_views = cfg.data.mvrecon.args.num_views
        self.num_input_views = cfg.data.mvrecon.args.num_input_views
        assert self.num_input_views == 1, "NMR currently only support 1 view as training"
        if cfg.debug.flag:
            self.scene_paths = self.scene_paths[:cfg.debug.debug_size]

    def __len__(self):
        return len(self.scene_paths)

    def __getitem__(self, index):
        scene_idx = index
        view_idx = random.sample(range(self.num_views), )
        target_views = np.array(list(set(range(self.num_views)) - set([view_idx])))
        scene_path = os.path.join(self.base_dir, self.scene_paths[scene_idx])
        # Load images
        images = [np.asarray(
            Image.open(os.path.join(scene_path, "image", f"{i:04d}.png"))
        ) for i in range(self.num_views)]
        images = np.stack(images, 0).astype(np.float32) / 255.
        input_image = np.transpose(images[view_idx], (2, 0, 1))
        # Load camera poses
        cameras = np.load(os.path.join(scene_path, "cameras.npz"))
        cameras = {k : v for k, v in cameras.items()}
        # Apply rotation matrix
        for i in range(self.num_views):
            cameras[f"world_mat_inv_{i}"] = self.rot_mat @ cameras[f"world_mat_inv_{i}"]
            cameras[f'world_mat_{i}'] = cameras[f'world_mat_{i}'] @ np.transpose(self.rot_mat)
        # Create rays
        rays = []
        width = images.shape[2]
        height = images.shape[1]
        xmap = np.linspace(-1, 1, width)
        ymap = np.linspace(-1, 1, height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        for i in range(self.num_views):
            cur_rays = np.stack((xmap, ymap, np.ones_like(xmap)), -1)
            cur_rays = transform_points(
                cur_rays, cameras[f"world_mat_inv_{i}"] @ cameras[f"camera_mat_inv_{i}"], translate=False
            )
            cur_rays = cur_rays[..., :3]
            cur_rays = cur_rays / np.linalg.norm(cur_rays, axis=-1, keepdims=True)
            rays.append(cur_rays)
        rays = np.stack(rays, axis=0).astype(np.float32)
        camera_pos = [cameras[f"world_mat_inv_{i}"][:3, -1] for i in range(self.num_views)]
        camera_pos = np.stack(camera_pos, axis=0).astype(np.float32)
        # Switch to canonical
        if self.canonical:
            canonical_extrinsic = cameras[f"world_mat_{view_idx}"].astype(np.float32)
            camera_pos = transform_points(camera_pos, canonical_extrinsic)
            rays = transform_points(rays, canonical_extrinsic, translate=False)
        rays_flat = np.reshape(rays[target_views], (-1, 3))
        pixels_flat = np.reshape(images[target_views], (-1, 3))
        cpos_flat = np.tile(np.expand_dims(camera_pos[target_views], 1), (1, width * height, 1))
        cpos_flat = np.reshape(cpos_flat, (len(target_views) * width * height, 3))
        num_points = rays_flat.shape[0]
        if not self.full_scale:
            replace = (num_points < self.points_per_item)
            sampled_idxs = np.random.choice(np.arange(num_points), size=(self.points_per_item, ), replace=replace)
            rays_sel = rays_flat[sampled_idxs]
            pixels_sel = pixels_flat[sampled_idxs]
            cpos_sel = cpos_flat[sampled_idxs]
        else:
            rays_sel = rays_flat
            pixels_sel = pixels_flat
            cpos_sel = cpos_flat
        result = {
            "input_images": torch.from_numpy(np.expand_dims(input_image, 0)),                  # [1, h, w, 3]
            "input_camera_pos": torch.from_numpy(np.expand_dims(camera_pos[view_idx], 0)),    # [1, 3]
            "input_rays": torch.from_numpy(np.expand_dims(rays[view_idx], 0)),          # [1, h, w, 3]
            "target_images": torch.from_numpy(pixels_sel),
            "target_camera_pos": torch.from_numpy(cpos_sel),
            "target_rays": torch.from_numpy(rays_sel),
            "scene_id": index
        }
        if self.canonical:
            result["transform"] = canonical_extrinsic   # [3, 4]
        return result

    def load_nmr(self):
        with open(os.path.join(self.base_dir, "metadata.yaml"), "r") as f:
            metadata = yaml.load(f, Loader=yaml.CLoader)
        class_ids = [entry["id"] for entry in metadata.values()]
        self.scene_paths = []
        for class_id in class_ids:
            with open(os.path.join(self.base_dir, class_id, f"softras_{self.split}.lst"), "r") as f:
                cur_scene_ids = f.readlines()
            cur_scene_ids = [scene_id.strip() for scene_id in cur_scene_ids if len(scene_id) > 1]
            cur_scene_paths = [os.path.join(class_id, scene_id) for scene_id in cur_scene_ids]
            self.scene_paths.extend(cur_scene_paths)
        self.num_scenes = len(self.scene_paths)