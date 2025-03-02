import copy
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch import Tensor

from model.build import MODEL_REGISTRY, BaseModel
from modules.build import build_module
from modules.layers.transformers import (TransformerEncoderLayer,
                                         TransformerSpatialEncoderLayer)
from modules.utils import (calc_pairwise_locs, disabled_train,
                           get_mixup_function, get_mlp_head, layer_repeat,
                           maybe_autocast)
from modules.weights import _init_weights_bert
from optim.utils import no_decay_param_group


def freeze_bn(m):
    '''Freeze BatchNorm Layers'''
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


def generate_fourier_features(pos, num_bands=10, max_freq=15, concat_pos=True, sine_only=False):
    # Input: B, N, C
    # Output: B, N, C'
    batch_size = pos.shape[0]
    device = pos.device

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.linspace(start=min_freq, end=max_freq, steps=num_bands, device=device)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos.unsqueeze(-1).repeat(1, 1, 1, num_bands) * freq_bands
    per_pos_features = torch.reshape(
        per_pos_features, [batch_size, -1, np.prod(per_pos_features.shape[2:])])
    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat(
            [pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)
    return per_pos_features


class FC(nn.Module):
    def __init__(self, in_size, out_size, pdrop=0., use_gelu=True):
        super(FC, self).__init__()
        self.pdrop = pdrop
        self.use_gelu = use_gelu

        self.linear = nn.Linear(in_size, out_size)

        if use_gelu:
            #self.relu = nn.Relu(inplace=True)
            self.gelu = nn.GELU()

        if pdrop > 0:
            self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        x = self.linear(x)

        if self.use_gelu:
            #x = self.relu(x)
            x = self.gelu(x)

        if self.pdrop > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, pdrop=0., use_gelu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, pdrop=pdrop, use_gelu=use_gelu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flat_mlp_size=512, flat_glimpses=1, flat_out_size=1024, pdrop=0.1):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flat_mlp_size,
            out_size=flat_glimpses,
            pdrop=pdrop,
            use_gelu=True
        )
        self.flat_glimpses = flat_glimpses

        self.linear_merge = nn.Linear(
            hidden_size * flat_glimpses,
            flat_out_size
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        return x_atted, att


class ObjColorEncoder(nn.Module):
    def __init__(self, hidden_size, dropout=0):
        super().__init__()
        self.ft_linear = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-12),
            nn.Dropout(dropout)
        )

    def forward(self, obj_colors):
        # obj_colors: (batch, nobjs, 3, 4)
        gmm_weights = obj_colors[..., :1]
        gmm_means = obj_colors[..., 1:]

        embeds = torch.sum(self.ft_linear(gmm_means) * gmm_weights, 2)
        return embeds


@MODEL_REGISTRY.register()
class OSE3D(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg.model

        self.vision_backbone_name = self.cfg.vision_backbone_name
        # spatial attention
        self.use_spatial_attn = self.cfg.use_spatial_attn
        # self object
        self.use_anchor = self.cfg.use_anchor
        # orientation embedding
        self.use_orientation = self.cfg.use_orientation

        # All objects:
        # -object feat
        # -object orientation feat
        # -object loc (loc + size, for spatial attention)
        # -object type embedding (normal object, self object (TBD: camera object))
        if self.use_anchor:
            # learnable self object feature and size
            self.anchor_feat = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden_size))
            self.anchor_size = nn.Parameter(torch.ones(1, 1, 3), requires_grad=False)
        if self.use_orientation:
            # learnable object orientation
            self.object_orientation_feat = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden_size))
            self.orientation_encoder = nn.Linear(self.cfg.fourier_size, self.cfg.hidden_size)
        self.object_type_embedding = nn.Embedding(2, embedding_dim=self.cfg.hidden_size)

        if self.vision_backbone_name == 'gt':
            self.obj_sem_encoder = nn.Linear(self.cfg.label_size, self.cfg.hidden_size)
            self.color_encoder = ObjColorEncoder(self.cfg.hidden_size)
        else: # GT Obj pcds
            self.obj_encoder = build_module('vision', self.cfg.vision)
            if self.cfg.vision.name == "PcdObjEncoder":
                self.obj_linear_projection = nn.Linear(self.cfg.vision.args.sa_mlps[-1][-1], self.cfg.hidden_size)
            elif self.cfg.vision.name == "PointBERTPcdObjEncoder":
                self.obj_linear_projection = nn.Linear(self.cfg.vision.args.trans_dim * 2, self.cfg.hidden_size)

        if self.cfg.use_spatial_attn:
            spatial_encoder_layer = TransformerSpatialEncoderLayer(
                self.cfg.hidden_size,
                nhead=self.cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=self.cfg.spatial_encoder.dim_feedforward,
                dropout=self.cfg.spatial_encoder.dropout,
                activation=self.cfg.spatial_encoder.activation,
                spatial_dim=self.cfg.spatial_encoder.spatial_dim,
                spatial_multihead=self.cfg.spatial_encoder.spatial_multihead,
                spatial_attn_fusion=self.cfg.spatial_encoder.spatial_attn_fusion,
            )
            self.spatial_encoder = layer_repeat(
                spatial_encoder_layer,
                self.cfg.spatial_encoder.num_layers
            )
        else:
            spatial_encoder_layer = TransformerEncoderLayer(
                self.cfg.hidden_size,
                nhead=self.cfg.spatial_encoder.num_attention_heads,
                dim_feedforward=self.cfg.spatial_encoder.dim_feedforward,
                dropout=self.cfg.spatial_encoder.dropout,
                activation=self.cfg.spatial_encoder.activation,
            )
            self.spatial_encoder = layer_repeat(
                spatial_encoder_layer,
                self.cfg.spatial_encoder.num_layers
            )

        if self.cfg.spatial_encoder.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.cfg.spatial_encoder.obj_loc_encoding == 'diff_all':
            num_loc_layers = self.cfg.spatial_encoder.num_layers
        loc_layer = nn.Sequential(
            nn.Linear(self.cfg.spatial_encoder.dim_loc, self.cfg.hidden_size),
            nn.LayerNorm(self.cfg.hidden_size),
        )
        self.loc_layers = layer_repeat(loc_layer, num_loc_layers)
        # Only initialize spatial encoder and loc layers
        self.spatial_encoder.apply(_init_weights_bert)
        self.loc_layers.apply(_init_weights_bert)

        # We can choose to aggregate object features into 1 token
        if self.cfg.attn_flat.use_attn_flat:
            self.attflat_visual = AttFlat(self.cfg.hidden_size,
                                        self.cfg.attn_flat.mcan_flat_mlp_size,
                                        self.cfg.attn_flat.mcan_flat_glimpses,
                                        self.cfg.attn_flat.mcan_flat_out_size,
                                        0.1)

        if self.use_anchor:
            torch.nn.init.normal_(self.anchor_feat, std=.02)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_gtpcd(self, data_dict):
        obj_pcd_feat = self.obj_linear_projection(self.obj_encoder(data_dict['obj_fts'])[0])
        return obj_pcd_feat

    def forward(self, data_dict):
        # Input:
        #   required keys:
        #       obj_fts: (B, N, M, 6)
        #       obj_masks: (B, N) 1 -- not masked, 0 -- masked
        #       obj_locs: (B, N, 6)
        #       -when use_anchor and use_orientation:
        #           anchor_orientation: (B, C)
        #           anchor_locs: (B, 3)
        #       -when us gt backbone (unused)
        #           obj_sems (CLIP)
        #           obj_colors

        if self.vision_backbone_name == 'gt':
            assert False
            object_feat = self.obj_embedding(data_dict['obj_sems'])
            object_feat = object_feat + self.color_encoder(data_dict['obj_colors'])
            object_mask = ~data_dict['obj_masks']
        else: # GT obj pcds
            if 'single_obj' in data_dict:   # get the token of single obj
                # TODO(lhxk) this method requires extra memory to store obj_fts, maybe we have better solution to avoid this operation
                obj_fts = data_dict['obj_fts'].clone()
                data_dict['obj_fts'] = data_dict['single_obj']
                data_dict['single_obj_token'] = self.forward_gtpcd(data_dict)
                data_dict['obj_fts'] = obj_fts
                return data_dict
            object_feat = self.forward_gtpcd(data_dict)
            object_mask = ~data_dict['obj_masks']

        B, N = object_feat.shape[:2]
        device = object_feat.device

        # All objects:
        # -object feat
        # -object orientation feat
        # -object loc (loc + size, for spatial attention)
        # -object type embedding (normal object, self object (TBD: camera object))
        # -object mask

        if self.use_orientation:
            object_orientation_feat = self.object_orientation_feat.expand(B, N, -1).to(device)
        # loc(3) + size(3)
        object_loc = data_dict['obj_locs']
        object_type_id = torch.zeros((B, N), dtype=torch.long, device=device)
        object_type_embedding = self.object_type_embedding(object_type_id)

        if self.use_anchor:
            anchor_feat = self.anchor_feat.expand(B, -1, -1).to(device)
            anchor_mask = torch.zeros((B, 1), device=device, dtype=bool)
            if self.use_orientation:
                # B, C -> B, 1, C
                anchor_orientation = data_dict['anchor_orientation'].unsqueeze(1).to(device)
                anchor_orientation_feat = self.orientation_encoder(generate_fourier_features(anchor_orientation))
            # TODO(jxma): we assume anchor loc only contains loc (B, 3)
            assert data_dict['anchor_locs'].shape[-1] == 3
            anchor_loc = torch.cat(
                (data_dict['anchor_locs'].unsqueeze(1), self.anchor_size.expand(B, -1, -1).to(device)), dim=-1)
            anchor_type_id = torch.ones((B, 1), dtype=torch.long, device=device)
            anchor_type_embedding = self.object_type_embedding(anchor_type_id)

            all_object_feat = torch.cat((anchor_feat, object_feat), dim=1)
            all_object_mask = torch.cat((anchor_mask, object_mask), dim=1)
            if self.use_orientation:
                all_object_orientation_feat = torch.cat((anchor_orientation_feat, object_orientation_feat), dim=1)
            all_object_loc = torch.cat((anchor_loc, object_loc), dim=1)
            all_object_type_embedding = torch.cat((anchor_type_embedding, object_type_embedding), dim=1)
        else:
            all_object_feat = object_feat
            all_object_mask = object_mask
            if self.use_orientation:
                all_object_orientation_feat = object_orientation_feat
            all_object_loc = object_loc
            all_object_type_embedding = object_type_embedding

        if self.use_orientation:
            all_object_feat = all_object_feat + all_object_orientation_feat + all_object_type_embedding
        else:
            all_object_feat = all_object_feat + all_object_type_embedding

        # call spatial encoder
        if self.cfg.use_spatial_attn:
            pairwise_locs = calc_pairwise_locs(
                all_object_loc[:, :, :3],
                all_object_loc[:, :, 3:],
                pairwise_rel_type=self.cfg.spatial_encoder.pairwise_rel_type,
                spatial_dist_norm=self.cfg.spatial_encoder.spatial_dist_norm,
                spatial_dim=self.cfg.spatial_encoder.spatial_dim
            )
        obj_embeds = all_object_feat
        with maybe_autocast(self, enabled=False):
            for i, pc_layer in enumerate(self.spatial_encoder):
                if self.cfg.spatial_encoder.obj_loc_encoding == 'diff_all':
                    query_pos = self.loc_layers[i](all_object_loc)
                    obj_embeds = obj_embeds + query_pos
                else:
                    query_pos = self.loc_layers[0](all_object_loc)
                    # TODO: obj_locs should be computed using object_feats, in votenet, how is that done?
                    if self.cfg.spatial_encoder.obj_loc_encoding == 'same_all':
                        obj_embeds = obj_embeds + query_pos
                    else:
                        if i == 0:
                            obj_embeds = obj_embeds + query_pos

                if self.cfg.use_spatial_attn:
                    obj_embeds, self_attn_matrices = pc_layer(
                        obj_embeds,
                        pairwise_locs,
                        tgt_key_padding_mask=all_object_mask)
                else:
                    obj_embeds, self_attn_matrices = pc_layer(
                        obj_embeds,
                        tgt_key_padding_mask=all_object_mask)

        all_object_feat = obj_embeds

        if self.cfg.attn_flat.use_attn_flat:
            all_object_feat, data_dict["oatt"] = self.attflat_visual(
                all_object_feat,
                all_object_mask
            )
        else:
            data_dict["oatt"] = None

        data_dict['obj_tokens'] = all_object_feat
        data_dict['obj_masks'] = ~all_object_mask

        return data_dict
