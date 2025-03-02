"""

    Created on 2023/8/9

    @author: Baoxiong Jia

"""

import torch
import torch.nn as nn

from Swin3D.models import Swin3DUNet
from MinkowskiEngine import SparseTensor
from modules.build import VISION_REGISTRY


@VISION_REGISTRY.register()
class PCDSwin3DEncoder(nn.Module):
    def __init__(self, cfg, depths, channels, num_heads, window_sizes, up_k, quant_size,
                 drop_path_rate=0.2, num_layers=4, num_classes=13, stem_transformer=False, upsample="deconv",
                 down_stride=2, knn_down=True, signal=True, in_channels=6, use_offset=False, fp16_mode=2,
                 **kwargs):
        super().__init__()

        # Swin3D hyperparameters
        self.signal = signal
        self.use_offset = use_offset
        self.backbone = Swin3DUNet(depths, channels, num_heads, window_sizes, quant_size, up_k=up_k,
                                   drop_path_rate=drop_path_rate, num_layers=num_layers, num_classes=num_classes,
                                   stem_transformer=stem_transformer, upsample=upsample,first_down_stride=down_stride,
                                   knn_down=knn_down, in_channels=in_channels, cRSE="XYZ_RGB", fp16_mode=fp16_mode)

    def forward(self, feats, xyz):
        # Swin3D preprocess
        device = feats.device
        coords = torch.cat([xyz[:, -1].unsqueeze(-1), xyz[:, :3]] ,dim=-1)
        feats = torch.cat([feats, xyz[:, :3]], dim=1)
        if self.signal:
            if feats.shape[1] > 3:
                if self.use_offset:
                    feats[:, -3:] = xyz[:, :3] - xyz[:, :3].int()
            sp = SparseTensor(feats.float(), coords.int(), device=device)
        else:
            sp = SparseTensor(torch.ones_like(feats).float(), coords.int(), device=device)
        colors = feats[:, 0:3] / 1.001
        coords_sp = SparseTensor(
            features=torch.cat([coords, colors], dim=1),
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager
        )
        swin3d_feats = self.backbone(sp, coords_sp)
        return swin3d_feats
