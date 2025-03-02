import math

import torch
import torch.nn as nn

from modules.layers.transformers import CrossAttentionLayer
from modules.utils import layer_repeat


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords):
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.pos_encoding = PositionalEncoding(num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, pos, rays):
        if len(rays.shape) == 4:
            batchsize, height, width, dims = rays.shape
            pos_enc = self.pos_encoding(pos.unsqueeze(1))
            pos_enc = pos_enc.view(batchsize, pos_enc.shape[-1], 1, 1)
            pos_enc = pos_enc.repeat(1, 1, height, width)
            rays = rays.flatten(1, 2)

            ray_enc = self.ray_encoding(rays)
            ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
            ray_enc = ray_enc.permute((0, 3, 1, 2))
            x = torch.cat((pos_enc, ray_enc), 1)
        else:
            pos_enc = self.pos_encoding(pos)
            ray_enc = self.ray_encoding(rays)
            x = torch.cat((pos_enc, ray_enc), -1)

        return x


class RayPredictor(nn.Module):
    def __init__(self, num_att_blocks=2, pos_start_octave=0, out_dims=3,
                 z_dim=768, input_mlp=False, output_mlp=True):
        super().__init__()

        if input_mlp:  # Input MLP added with OSRT improvements
            self.input_mlp = nn.Sequential(
                nn.Linear(180, 360),
                nn.ReLU(),
                nn.Linear(360, 180))
        else:
            self.input_mlp = None

        self.query_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                        ray_octaves=15)
        self.transformer_layers = layer_repeat(
            CrossAttentionLayer(d_model=180, nhead=12, dim_feedforward=z_dim * 2,
                                activation="gelu", k_dim=z_dim, v_dim=z_dim, prenorm=True),
            num_att_blocks
        )

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(180, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

    def forward(self, z, x, rays):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        """
        queries = self.query_encoder(x, rays)
        if self.input_mlp is not None:
            queries = self.input_mlp(queries)
        for layer in self.transformer_layers:
            queries, _ = layer(queries, z)
        output = queries
        if self.output_mlp is not None:
            output = self.output_mlp(output)
        return output