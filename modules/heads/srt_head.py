import torch
import torch.nn as nn

from modules.build import HEADS_REGISTRY
from modules.layers.srt import RayPredictor, PositionalEncoding
from modules.layers.transformers import CrossAttentionLayer


@HEADS_REGISTRY.register()
class SRTDecoder(nn.Module):
    def __init__(self, cfg, num_att_blocks=2, hidden_dim=768, pos_start_octave=0):
        super().__init__()
        self.allocation_transformer = RayPredictor(num_att_blocks=num_att_blocks,
                                                   pos_start_octave=pos_start_octave,
                                                   z_dim=hidden_dim, input_mlp=True, output_mlp=False)
        self.render_mlp = nn.Sequential(
            nn.Linear(180, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 3),
        )

    def forward(self, features, query_points, rays):
        """
        SRT decoder
        Args:
            features:       (B, num_patches, hidden_dim)
            query_points:   (B, num_rays, 3)
            rays:           (B, num_rays, 3)

        Returns:
        """
        x = self.allocation_transformer(features, query_points, rays)
        pixels = self.render_mlp(x)
        return torch.sigmoid(pixels)