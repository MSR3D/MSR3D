import einops
import torch
import torch.nn as nn

from modules.build import VISION_REGISTRY
from modules.third_party.pointbert.pointbert import PointTransformer


@VISION_REGISTRY.register()
class PointBERTPcdObjEncoder(nn.Module):
    def __init__(self, 
                 cfg, 
                 trans_dim, 
                 depth, 
                 drop_path_rate, 
                 cls_dim, 
                 num_heads, 
                 group_size, 
                 num_group, 
                 encoder_dims, 
                 add_RGB=True, 
                 path=None, 
                 freeze=False) -> None:
        super().__init__()

        self.pcd_encoder = PointTransformer(
            trans_dim=trans_dim, 
            depth=depth, 
            drop_path_rate=drop_path_rate, 
            cls_dim=cls_dim, 
            num_heads=num_heads, 
            group_size=group_size, 
            num_group=num_group, 
            encoder_dims=encoder_dims, 
            add_RGB=add_RGB
        )

        if path is not None:
            self.pcd_encoder.load_model_from_ckpt(path)

        self.freeze = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def freeze_bn(self, m):
        '''Freeze BatchNorm Layers'''
        for layer in m.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, obj_pcds, obj_locs=None, obj_masks=None, obj_sem_masks=None, **kwargs):
        batch_size, num_objs, _, _ = obj_pcds.size()
        if self.freeze:
            self.freeze_bn(self.pcd_encoder)
            with torch.no_grad():
                obj_embeds = self.pcd_encoder(
                    einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
                )
                obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
                obj_embeds = obj_embeds.detach()
        else:
            obj_embeds = self.pcd_encoder(
                einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
            )
            obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
        
        return obj_embeds, obj_embeds   # simplified here, second term should be the features after projector
