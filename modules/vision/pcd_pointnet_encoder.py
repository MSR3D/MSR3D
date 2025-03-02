import einops
import torch
from torch import nn

from modules.build import VISION_REGISTRY
from modules.layers.pointnet import PointNetPP
from modules.utils import get_mlp_head


@VISION_REGISTRY.register()
class PcdObjEncoder(nn.Module):
    def __init__(self,
                 cfg,
                 sa_n_points=[32, 16, None],
                 sa_n_samples=[32, 32, None],
                 sa_radii=[0.2, 0.4, None],
                 sa_mlps=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]],
                 dropout=0.1,
                 path=None,
                 freeze=False):
        super().__init__()

        self.pcd_net = PointNetPP(
            sa_n_points=sa_n_points,
            sa_n_samples=sa_n_samples,
            sa_radii=sa_radii,
            sa_mlps=sa_mlps,
        )

        self.obj3d_clf_pre_head = get_mlp_head(sa_mlps[-1][-1], 384, 607, dropout=0.3)

        self.dropout = nn.Dropout(dropout)

        if path is not None:
            self.load_state_dict(torch.load(path), strict=False)

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
            self.freeze_bn(self.pcd_net)
            with torch.no_grad():
                obj_embeds = self.pcd_net(
                    einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
                )
                obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)
                obj_embeds = obj_embeds.detach()
        else:
            obj_embeds = self.pcd_net(
                einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
            )
            obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # # TODO: due to the implementation of PointNetPP, this way consumes less GPU memory
        # obj_embeds = []
        # for i in range(batch_size):
        #     obj_embeds.append(self.pcd_net(obj_pcds[i]))
        # obj_embeds = torch.stack(obj_embeds, 0)
        # obj_embeds = self.dropout(obj_embeds)
        # freeze

        # sem logits
        obj_sem_cls = self.obj3d_clf_pre_head(obj_embeds)
        return obj_embeds, obj_sem_cls
