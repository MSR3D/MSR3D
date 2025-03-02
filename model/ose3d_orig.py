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
from modules.layers.pointnet import PointNetPP
from modules.utils import layer_repeat, maybe_autocast
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
    per_pos_features = pos.unsqueeze(-1).repeat(1, 1, 1, 10) * freq_bands
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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadAttentionSpatial(nn.Module):
    def __init__(
        self, d_model, n_head, dropout=0.1, spatial_multihead=True, spatial_dim=5,
        spatial_attn_fusion='mul',
    ):
        super().__init__()
        assert d_model % n_head == 0, 'd_model: %d, n_head: %d' %(d_model, n_head)

        self.n_head = n_head
        self.d_model = d_model
        self.d_per_head = d_model // n_head
        self.spatial_multihead = spatial_multihead
        self.spatial_dim = spatial_dim
        self.spatial_attn_fusion = spatial_attn_fusion

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.spatial_n_head = n_head if spatial_multihead else 1
        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            self.pairwise_loc_fc = nn.Linear(spatial_dim, self.spatial_n_head)
        elif self.spatial_attn_fusion == 'ctx':
            self.pairwise_loc_fc = nn.Linear(spatial_dim, d_model)
        elif self.spatial_attn_fusion == 'cond':
            self.lang_cond_fc = nn.Linear(d_model, self.spatial_n_head * (spatial_dim + 1))
        else:
            raise NotImplementedError('unsupported spatial_attn_fusion %s' % (self.spatial_attn_fusion))

    def forward(self, q, k, v, pairwise_locs, layer_depth=-1, key_padding_mask=None, txt_embeds=None, temparature=[1, 0.7, 0.5, 0.3]):
        residual = q
        q = einops.rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
        k = einops.rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
        v = einops.rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)
        attn = torch.einsum('hblk,hbtk->hblt', q, k) / np.sqrt(q.shape[-1])

        if self.spatial_attn_fusion in ['mul', 'bias', 'add']:
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t h -> h b l t')
            if self.spatial_attn_fusion == 'mul':
                loc_attn = F.relu(loc_attn)
            if not self.spatial_multihead:
                loc_attn = einops.repeat(loc_attn, 'h b l t -> (h nh) b l t', nh=self.n_head)
        elif self.spatial_attn_fusion == 'ctx':
            loc_attn = self.pairwise_loc_fc(pairwise_locs)
            loc_attn = einops.rearrange(loc_attn, 'b l t (h k) -> h b l t k', h=self.n_head)
            loc_attn = torch.einsum('hblk,hbltk->hblt', q, loc_attn) / np.sqrt(q.shape[-1])
        elif self.spatial_attn_fusion == 'cond':
            spatial_weights = self.lang_cond_fc(residual) #  + txt_embeds.unsqueeze(1))   # NOTE: original add, now remove
            spatial_weights = einops.rearrange(spatial_weights, 'b l (h d) -> h b l d', h=self.spatial_n_head, d=self.spatial_dim+1)
            if self.spatial_n_head == 1:
                spatial_weights = einops.repeat(spatial_weights, '1 b l d -> h b l d', h=self.n_head)
            spatial_bias = spatial_weights[..., :1]
            spatial_weights = spatial_weights[..., 1:]
            loc_attn = torch.einsum('hbld,bltd->hblt', spatial_weights, pairwise_locs) + spatial_bias
            loc_attn = torch.sigmoid(loc_attn)
        if key_padding_mask is not None:
            mask = einops.repeat(key_padding_mask, 'b t -> h b l t', h=self.n_head, l=q.size(2))
            attn = attn.masked_fill(mask, -np.inf)
            if self.spatial_attn_fusion in ['mul', 'cond']:
                loc_attn = loc_attn.masked_fill(mask, 0)
            else:
                loc_attn = loc_attn.masked_fill(mask, -np.inf)

        if self.spatial_attn_fusion == 'add':
            fused_attn = (torch.softmax(attn, 3) + torch.softmax(loc_attn, 3)) / 2
        else:
            if self.spatial_attn_fusion in ['mul', 'cond']:
                # fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) * temparature[layer_depth] + attn * (2 - temparature[layer_depth])
                fused_attn = torch.log(torch.clamp(loc_attn, min=1e-6)) + attn
            else:
                fused_attn = loc_attn + attn
            fused_attn = torch.softmax(fused_attn, 3)

        assert torch.sum(torch.isnan(fused_attn) == 0), "NaN appears"# print(fused_attn)

        output = torch.einsum('hblt,hbtv->hblv', fused_attn, v)
        output = einops.rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, fused_attn


class TransformerSpatialEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 spatial_multihead=True, spatial_dim=5, spatial_attn_fusion='mul', layer_depth=-1):

        super(TransformerSpatialEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttentionSpatial(
            d_model, nhead, dropout=dropout,
            spatial_multihead=spatial_multihead,
            spatial_dim=spatial_dim,
            spatial_attn_fusion=spatial_attn_fusion,
        )

        self.layer_depth = layer_depth

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt, tgt_pairwise_locs,
        tgt_key_padding_mask: Optional[Tensor] = None, temparature=[1, 0.7, 0.5, 0.3]):

        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_pairwise_locs, self.layer_depth,
            key_padding_mask=tgt_key_padding_mask,
            txt_embeds=None, temparature=temparature)

        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt = self.norm2(tgt + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        ))

        return tgt


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", layer_depth=-1):

        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.layer_depth = layer_depth

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt,
        tgt_key_padding_mask: Optional[Tensor] = None):

        tgt2, _ = self.self_attn(tgt, tgt, tgt,
            key_padding_mask=tgt_key_padding_mask)

        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt = self.norm2(tgt + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        ))

        return tgt


class TransformerSpatialEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if self.config.spatial_enc:
            encoder_class = TransformerSpatialEncoderLayer
            kwargs = {
                'spatial_dim': config.spatial_dim,
                'spatial_multihead': config.spatial_multihead,
                'spatial_attn_fusion': config.spatial_attn_fusion,
            }
        else:
            encoder_class = TransformerEncoderLayer
            kwargs = {}

        encoder_layer = encoder_class(
            config.hidden_size, config.num_attention_heads,
            dim_feedforward=2048, dropout=0.1, activation='gelu', **kwargs
        )
        self.layers = _get_clones(encoder_layer, config.num_layers)
        for i, layer in enumerate(self.layers):
            layer.layer_depth = i
        print(f"using {config.num_layers} layer(s) of spatial attention")

        loc_layer = nn.Sequential(
            nn.Linear(config.dim_loc, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
        )

        if self.config.obj_loc_encoding in ['same_0', 'same_all']:
            num_loc_layers = 1
        elif self.config.obj_loc_encoding == 'diff_all':
            num_loc_layers = config.num_layers
        self.loc_layers = _get_clones(loc_layer, num_loc_layers)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calc_pairwise_locs(self, obj_centers, obj_whls, eps=1e-10, pairwise_rel_type='center'):
        if pairwise_rel_type == 'mlp':
            obj_locs = torch.cat([obj_centers, obj_whls], 2)
            pairwise_locs = torch.cat(
                [einops.repeat(obj_locs, 'b l d -> b l x d', x=obj_locs.size(1)),
                einops.repeat(obj_locs, 'b l d -> b x l d', x=obj_locs.size(1))],
                dim=3
            )
            return pairwise_locs

        pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
            - einops.repeat(obj_centers, 'b l d -> b 1 l d')
        pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps) # (b, l, l)
        if self.config.spatial_dist_norm:
            max_dists = torch.max(pairwise_dists.view(pairwise_dists.size(0), -1), dim=1)[0]
            norm_pairwise_dists = pairwise_dists / einops.repeat(max_dists, 'b -> b 1 1')
        else:
            norm_pairwise_dists = pairwise_dists

        if self.config.spatial_dim == 1:
            return norm_pairwise_dists.unsqueeze(3)

        pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2]**2, 3)+eps)
        if pairwise_rel_type == 'center':
            pairwise_locs = torch.stack(
                [norm_pairwise_dists, pairwise_locs[..., 2]/pairwise_dists,
                pairwise_dists_2d/pairwise_dists, pairwise_locs[..., 1]/pairwise_dists_2d,
                pairwise_locs[..., 0]/pairwise_dists_2d],
                dim=3
            )
        elif pairwise_rel_type == 'vertical_bottom':
            bottom_centers = torch.clone(obj_centers)
            bottom_centers[:, :, 2] -= obj_whls[:, :, 2]
            bottom_pairwise_locs = einops.repeat(bottom_centers, 'b l d -> b l 1 d') \
                - einops.repeat(bottom_centers, 'b l d -> b 1 l d')
            bottom_pairwise_dists = torch.sqrt(torch.sum(bottom_pairwise_locs**2, 3) + eps) # (b, l, l)
            bottom_pairwise_dists_2d = torch.sqrt(torch.sum(bottom_pairwise_locs[..., :2]**2, 3)+eps)
            pairwise_locs = torch.stack(
                [norm_pairwise_dists,
                bottom_pairwise_locs[..., 2]/bottom_pairwise_dists,
                bottom_pairwise_dists_2d/bottom_pairwise_dists,
                pairwise_locs[..., 1]/pairwise_dists_2d,
                pairwise_locs[..., 0]/pairwise_dists_2d],
                dim=3
            )

        if self.config.spatial_dim == 4:
            pairwise_locs = pairwise_locs[..., 1:]
        return pairwise_locs

    def forward(
        self, obj_embeds, obj_locs, obj_masks,
        output_attentions=False, output_hidden_states=False, temparature=[1, 0.7, 0.5, 0.3]
    ):
        if self.config.spatial_enc:
            pairwise_locs = self.calc_pairwise_locs(
                obj_locs[:, :, :3], obj_locs[:, :, 3:],
                pairwise_rel_type=self.config.pairwise_rel_type
            )

        out_embeds = obj_embeds
        all_hidden_states = [out_embeds]
        all_self_attn_matrices, all_cross_attn_matrices = [], []
        for i, layer in enumerate(self.layers):
            if self.config.obj_loc_encoding == 'diff_all':

                query_pos = self.loc_layers[i](obj_locs)
                out_embeds = out_embeds + query_pos
            else:
                query_pos = self.loc_layers[0](obj_locs)
                # TODO: obj_locs should be computed using object_feats, in votenet, how is that done?
                if self.config.obj_loc_encoding == 'same_all':
                    out_embeds = out_embeds + query_pos
                else:
                    if i == 0:
                        out_embeds = out_embeds + query_pos

            if self.config.spatial_enc:
                out_embeds = layer(
                    out_embeds, pairwise_locs,
                    tgt_key_padding_mask=obj_masks, temparature=temparature
                )
            else:
                out_embeds = layer(
                    out_embeds, tgt_key_padding_mask=obj_masks
                )

            all_hidden_states.append(out_embeds)

        outs = {
            'obj_embeds': out_embeds,
        }
        if output_hidden_states:
            outs['all_hidden_states'] = all_hidden_states
        return outs


class PcdObjEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pcd_net = PointNetPP(
            sa_n_points=config.sa_n_points,
            sa_n_samples=config.sa_n_samples,
            sa_radii=config.sa_radii,
            sa_mlps=config.sa_mlps,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, obj_pcds):
        batch_size, num_objs, _, _ = obj_pcds.size()
        obj_embeds = self.pcd_net(
            einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
        )
        obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # # TODO: due to the implementation of PointNetPP, this way consumes less GPU memory
        # obj_embeds = []
        # for i in range(batch_size):
        #     obj_embeds.append(self.pcd_net(obj_pcds[i]))
        # obj_embeds = torch.stack(obj_embeds, 0)

        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds


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
class OSE3DORIG(BaseModel):
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
            self.obj_encoder = PcdObjEncoder(self.cfg.obj_encoder)
            self.obj_linear_projection = nn.Linear(self.cfg.obj_encoder.sa_mlps[-1][-1], self.cfg.hidden_size)

        spatial_enc_config = EasyDict(self.cfg.spatial_encoder)
        spatial_enc_config.hidden_size = self.cfg.hidden_size
        spatial_enc_config.num_attention_heads = 8
        spatial_enc_config.spatial_enc = self.use_spatial_attn
        self.spatial_encoder = TransformerSpatialEncoder(spatial_enc_config)

        # We can choose to aggregate object features into 1 token
        if self.cfg.attn_flat.use_attn_flat:
            self.attflat_visual = AttFlat(self.cfg.hidden_size,
                                        self.cfg.attn_flat.mcan_flat_mlp_size,
                                        self.cfg.attn_flat.mcan_flat_glimpses,
                                        self.cfg.attn_flat.mcan_flat_out_size,
                                        0.1)

        if self.use_anchor:
            torch.nn.init.normal_(self.anchor_feat, std=.02)

        if self.cfg.obj_encoder.pretrain and self.vision_backbone_name == 'gtpcd':
            self._load_pretrained_PNPP()
            freeze_bn(self.obj_encoder)
            for p in self.obj_encoder.parameters():
                p.requires_grad = False

    def _load_pretrained_PNPP(self):
        sd = torch.load(self.cfg.obj_encoder.pretrain)
        self.obj_encoder.load_state_dict(sd, strict=False)
        # new_sd = {}
        # for (k, v) in sd.items():
        #     if 'obj_encoder' in k:
        #         new_k = '.'.join(k.split('.')[1 : ])
        #         new_sd[new_k] = v
        # self.obj_encoder.load_state_dict(new_sd)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward_gtpcd(self, data_dict):
        obj_pcd_feat = self.obj_linear_projection(self.obj_encoder(data_dict['obj_fts']))
        return obj_pcd_feat

    def forward(self, data_dict):
        # Input:
        #   required keys:
        #       obj_fts: (B, N, C)
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

        with maybe_autocast(self, enabled=False):
            out_embeds = self.spatial_encoder(
                all_object_feat,
                all_object_loc,
                all_object_mask,
                temparature=[1, 1, 1, 1])
        all_object_feat = out_embeds['obj_embeds']

        if self.cfg.attn_flat.use_attn_flat:
            all_object_feat, data_dict['oatt'] = self.attflat_visual(
                all_object_feat,
                all_object_mask
            )
        else:
            data_dict['oatt'] = None

        data_dict['obj_tokens'] = all_object_feat
        data_dict['obj_masks'] = ~all_object_mask

        return data_dict