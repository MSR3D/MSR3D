import math

import torch
import torch.nn as nn

from modules.build import VISION_REGISTRY
from modules.layers.srt import RayEncoder
from modules.layers.transformers import TransformerEncoderLayer
from modules.utils import layer_repeat


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim
        if odim is None:
            odim = 2 * hdim
        idim, hdim, odim = int(idim), int(hdim), int(odim)
        conv_kwargs = {"bias": False, "kernel_size": (3, 3), "padding": 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


@VISION_REGISTRY.register()
class SRTEncoder(nn.Module):
    def __init__(self, cfg, num_conv_blocks=3, num_att_blocks=5, pos_start_octave=0, hidden_dim=768,
                 num_heads=12, dim_head=64, mlp_dim=1536, scale_embeddings=False):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave, ray_octaves=15)
        ray_image_dim = 183
        cur_dim = 96
        conv_blocks = [SRTConvBlock(ray_image_dim, hdim=cur_dim)]
        cur_dim *= 2
        for i in range(num_conv_blocks):
            conv_blocks.append(SRTConvBlock(idim=cur_dim))
            cur_dim *= 2
        self.conv_blocks = nn.Sequential(*conv_blocks)
        cur_dim, hidden_dim = int(cur_dim), int(hidden_dim)
        self.per_patch_linear = nn.Conv2d(cur_dim, hidden_dim, kernel_size=1)
        self.num_att_blocks = num_att_blocks
        self.transformer_layers = layer_repeat(
            TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=mlp_dim, activation="gelu", prenorm=True),
            num_att_blocks
        )

    def forward(self, images, camera_pos, rays, **kwargs):
        """
        SRT multiview image encoder
        Args:
            images:     (B, N_views, 3, H, W)
            camera_pos: (B, N_views, 3)
            rays:       (B, N_views, H, W, 3)
            **kwargs:

        Returns:
            (B, N_views * N_patches, hidden_dim)
        """
        B, N_v = images.shape[:2]
        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        ray_enc = self.ray_encoder(camera_pos, rays)    # B * N_v, ray_dim, H, W
        x = torch.cat((x, ray_enc), 1)                  # B * N_v, 3+ray_dim, H, W
        x = self.conv_blocks(x)                         # B * N_v, conv_out_dim, P_h, P_w
        x = self.per_patch_linear(x)                    # B * N_v, hidden_dim, P_h, P_w
        x = x.flatten(2, 3).permute(0, 2, 1)            # B * N_v, P_h * P_w, hidden_dim

        N_p, N_c = x.shape[1:]
        x = x.reshape(B, N_v * N_p, N_c)
        for layer in self.transformer_layers:
            x, _ = layer(x)
        return x
