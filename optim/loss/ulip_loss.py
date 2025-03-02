'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch.nn as nn
import torch.nn.functional as F
from optim.loss.loss import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        # TODO: moved the gather to trainer as the accelerator is there
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        labels = outputs['labels']

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # TODO: potential problem because of total-batch x total-batch
        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed.t()

        loss = (F.cross_entropy(logits_per_pc_text, labels) + F.cross_entropy(logits_per_text_pc, self.labels)) / 2 \
               + (F.cross_entropy(logits_per_pc_image, labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        return loss