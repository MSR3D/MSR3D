import logging
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from einops import rearrange
from transformers import (Blip2Model, Blip2QFormerConfig, Blip2QFormerModel,
                          Blip2VisionConfig, Blip2VisionModel)

from modules.build import VISION_REGISTRY

logger = logging.getLogger(__name__)


def simple_conv_and_linear_weights_init(m):
    if type(m) in [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        simple_linear_weights_init(m)


def simple_linear_weights_init(m):
    if type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


@VISION_REGISTRY.register()
class Backbone2D(nn.Module):
    def __init__(self,
                 cfg,
                 backbone_name,
                 backbone_pretrain_dataset,
                 pooling=None,
                 flat_output=True,
                 use_pretrain=True):
        super().__init__()
        self.flat_output = flat_output
        backbone_name = backbone_name
        backbone_pretrain_dataset = backbone_pretrain_dataset
        backbone_use_pretrain = use_pretrain

        init_func = globals().get('_'.join([backbone_name, backbone_pretrain_dataset]))

        if init_func and callable(init_func):
            self.backbone = init_func(pretrained=backbone_use_pretrain)
        else:
            raise NotImplementedError(f'{backbone_name} + {backbone_pretrain_dataset} does not exist.')

        self.pooling = pooling
        if self.pooling:
            if self.pooling == 'avg':
                self.pooling_layers = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=(1,1)),
                        nn.Flatten()
                )
                self.out_channels = self.backbone.out_channels
            elif self.pooling == 'conv':
                self.pooling_layers = nn.Sequential(
                    nn.Conv2d(self.backbone.out_channels, 64, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 1),
                    nn.Flatten()
                )
                self.pooling_layers.apply(simple_conv_and_linear_weights_init)
                # FIXME(jxma): hard-coded
                self.out_channels = 32 * 7 * 7
            elif self.pooling == 'attn' or self.pooling == 'attention':
                self.visual_attention = nn.Sequential(
                    nn.Conv2d(self.backbone.out_channels, self.backbone.out_channels, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.backbone.out_channels, self.backbone.out_channels, 1),
                )
                self.visual_attention.apply(simple_conv_and_linear_weights_init)
                def _attention_pooling(x):
                    B, C, H, W = x.size()
                    attn = self.visual_attention(x)
                    attn = attn.view(B, C, -1)
                    x = x.view(B, C, -1)
                    attn = attn.softmax(dim=-1)
                    x = torch.einsum('b c n, b c n -> b c', x, x)
                    return x
                self.pooling_layers = _attention_pooling
                self.out_channels = self.backbone.out_channels
            else:
                raise NotImplementedError(f'do not support {self.pooling} pooling')
            print(f'Backbone2D: {self.pooling} pooling layer is used.', flush=True)
        else:
            print('Backbone2D: no pooling layer is used.', flush=True)
            self.out_channels = self.backbone.out_channels

    def forward(self, x):
        if self.pooling:
            x = self.backbone(x, flat_output=False)
            x = self.pooling_layers(x).unsqueeze(1)
            return x
        else:
            return self.backbone(x, flat_output=self.flat_output)

# 32x768
@VISION_REGISTRY.register()
class BLIP2Backbone(nn.Module):
    def __init__(self,
                 cfg,
                 model_name='Salesforce/blip2-opt-2.7b',
                 use_pretrain=True):
        super().__init__()

        if use_pretrain:
            blip2 = Blip2Model.from_pretrained(model_name)
            del blip2.language_model
            self.vision_model = blip2.vision_model
            self.qformer = blip2.qformer
            self.qformer_query = blip2.query_tokens
        else:
            self.vision_model = Blip2VisionModel(Blip2VisionConfig())
            self.qformer = Blip2QFormerModel(Blip2QFormerConfig())
            # FIXME: we use 32 query tokens by default
            self.qformer_query = nn.Parameter(torch.FloatTensor(
                1, 32, self.qformer.config.hidden_size).uniform_(-0.5, 0.5))

        self.out_channels = 768

    def forward(self, x):
        B = x.size(0)
        image_embs = self.vision_model(x).last_hidden_state
        image_embs = self.qformer(self.qformer_query.expand(B, -1, -1),
                                  encoder_hidden_states=image_embs).last_hidden_state
        return image_embs

class _Wrapper(nn.Module):

    def __init__(self, model, tag):
        super().__init__()
        self.model = model
        self.tag = tag
        if 'convnext' in tag:
            self.out_channels = 1024
        elif 'swin' in tag:
            self.out_channels = 1024
        elif 'vit' in tag:
            self.out_channels = 768
        elif 'resnet' in tag:
            self.out_channels = 2048
        else:
            raise NotImplementedError

    def forward(self, x, flat_output=False):
        feat = self.model.forward_features(x)
        if 'swin' in self.tag:
            feat = rearrange(feat, 'b h w c -> b c h w')
        if 'vit_base_32_timm_laion2b' in self.tag or 'vit_base_32_timm_openai' in self.tag:
            # TODO: [CLS] is prepended to the patches.
            feat = rearrange(feat[:, 1:], 'b (h w) c -> b c h w', h=7)
        if flat_output:
            feat = rearrange(feat, 'b c h w -> b (h w) c')
        return feat

# 1024x7x7 or 49x1024
def convnext_base_in1k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'convnext_base',
        pretrained=pretrained
    ), 'convnext_base_in1k')

# 1024x7x7 or 49x1024
def convnext_base_in22k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'convnext_base_in22k',
        pretrained=pretrained
    ), 'convnext_base_in22k')

# 1024x7x7 or 49x1024
# @register('convnext_base_laion400m')
# def convnext_base_in22k(pretrained=False, **kwargs):
#     model = timm.create_model(
#         'convnext_base',
#         pretrained=False
#     )
#     if pretrained:
#         model.load_state_dict(torch.load(os.path.join(
#             os.path.dirname(__file__),
#             '../../assets/weights',
#             'convnext_base_224_laion400m.pth'), map_location='cpu'), strict=False)
#         logger.info('Pretrained LAION-400m convnext-base loaded.')
#     return _Wrapper(model, 'convnext_base_laion400m')

# 1024x7x7 or 49x1024
def convnext_base_laion2b(pretrained=False, **kwargs):
    m = timm.create_model(
        'convnext_base.clip_laion2b',
        pretrained=pretrained
    )
    if kwargs.get('reset_clip_s2b2'):
        logger.info('Resetting the last conv layer of convnext-base to random init.')
        s = m.state_dict()
        for i in s.keys():
            if 'stages.3.blocks.2' in i and ('weight' in i or 'bias' in i):
                s[i].normal_()
        m.load_state_dict(s, strict=True)

    return _Wrapper(m, 'convnext_base_laion2b')

# 1024x7x7 or 49x1024
def swin_base_in1k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=pretrained
    ), 'swin_base_timm_in1k')

# 1024x7x7 or 49x1024
def swin_base_in22k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'swin_base_patch4_window7_224_in22k',
        pretrained=pretrained
    ), 'swin_base_timm_in22k')

# 768x7x7 or 49x768
def vit_b_32_laion2b(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'vit_base_patch32_clip_224.laion2b',
        pretrained=pretrained
    ), 'vit_base_32_timm_laion2b')

# 768x7x7 or 49x768
def vit_b_32_openai(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'vit_base_patch32_clip_224.openai',
        pretrained=pretrained
    ), 'vit_base_32_timm_openai')

# 2048x7x7 or 49x2048
def resnet_50_in1k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'resnet50.gluon_in1k',
        pretrained=pretrained
    ), 'resnet50_timm_in1k')

if __name__ == '__main__':
    # download all the models
    import torch
    torch.hub.set_dir('/scratch/ml/maxiaojian/hf_model_cache')
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/ml/maxiaojian/hf_model_cache'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/scratch/ml/maxiaojian/hf_model_cache'
    convnext_base_in1k(pretrained=True)
    print('++++++')
    convnext_base_in22k(pretrained=True)
    print('++++++')
    convnext_base_laion2b(pretrained=True)
    print('++++++')
    swin_base_in1k(pretrained=True)
    print('++++++')
    swin_base_in22k(pretrained=True)
    print('++++++')
    vit_b_32_laion2b(pretrained=True)
    print('++++++')
    vit_b_32_openai(pretrained=True)
    print('++++++')
    Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b')
    print('++++++')