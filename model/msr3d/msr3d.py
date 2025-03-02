"""

    Created on 2024/9/26

    @author: Xiongkun Linghu

    @acknowledgement: This code is adapted from LEO(An Embodied Generalist Agent in 3D World).

"""
import contextlib
import logging

import clip
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List, Optional, Union
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput

from data.data_utils import pad_tensors, find_subsequence, path_verify, save_to_json
from model.build import MODEL_REGISTRY, build_model
from modules.build import build_module
from modules.layers.transformers import GPT2Model, TransformerDecoderLayer
from modules.utils import layer_repeat, maybe_autocast
from IPython import embed
from torch.nn.functional import pad
import os

logger = logging.getLogger(__name__)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@MODEL_REGISTRY.register()
class MSR3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg_raw = cfg
        if hasattr(cfg, 'model'):
            cfg = cfg.model
        self.cfg = cfg

        # visual prompter
        self.visual_prompter = build_model(cfg.prompter)
        if cfg.prompter.model.vision.args.freeze:
            self.visual_prompter.obj_encoder.train = disabled_train

        # LLM
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(cfg.llm.cfg_path, use_fast=False,
                                                            truncation_side=cfg.llm.truncation_side)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # add special tokens for LLM
        self.image_placeholder = '图'
        self.object_placeholder = '物'
        self.scene_placeholder = '景'
        special_tokens = [self.image_placeholder, self.object_placeholder, self.scene_placeholder]
        special_tokens = special_tokens+self.llm_tokenizer.additional_special_tokens[len(special_tokens):]
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens}) 
        self.image_token_len = 16    # 16 tokens per image
        self.object_token_len = 8    # 8 tokens per object
        self.scene_token_len = cfg.prompter.model.get('scene_token_len', 61)    # 60 tokens per scene
        print('!!! scene token len !!!', self.scene_token_len)
        
        self.llm_model = LlamaForCausalLM.from_pretrained(cfg.llm.cfg_path, torch_dtype=torch.float16)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.max_context_len = cfg.llm.max_context_len

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model.train = disabled_train
        logger.info("Freeze LLM")

        self.llm_proj = nn.Linear(
            cfg.prompter.model.hidden_size, self.llm_model.config.hidden_size
        )

        self.action_transformer = None

        # image encoder
        self.image_encoder = build_module('vision', cfg.vision_2d)
        if self.cfg.vision_2d.freeze:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            self.image_encoder.eval()
            self.image_encoder.train = disabled_train
            logger.info("Freeze 2D backbone")
        self.llm_proj_img = nn.Linear(
            self.image_encoder.out_channels, self.llm_model.config.hidden_size
        )

        # LoRA-based LLM fine-tuning
        if cfg.llm.lora.flag:
            lora_config = LoraConfig(
                r=cfg.llm.lora.rank,
                lora_alpha=cfg.llm.lora.alpha,
                target_modules=cfg.llm.lora.target_modules,
                lora_dropout=cfg.llm.lora.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config=lora_config)

        self.prompt = cfg.llm.prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors='pt')
        self.prompt_length = int(prompt_tokens.attention_mask.sum(1)[0])

        self.max_context_len = cfg.llm.max_context_len
        self.max_out_len = cfg.llm.max_out_len
        self._lemmatizer = None

        # additional text x multi-modal tokens fusion
        self.clip_fusion = cfg.llm.clip_fusion
        self.clip_model = clip.load('RN50', device='cpu')[0].eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        self.clip_model.train = disabled_train
        clip_out_dim = 1024
        self.clip_proj = nn.Linear(clip_out_dim, self.llm_model.config.hidden_size)

        print(f"EmbodiedSolver built with {self.show_n_params(self.parameters())} parameters", flush=True)
        print(f"{self.show_n_params(self.get_opt_params())} learnable parameters", flush=True)
        self.log_opt_params()


    def log_opt_params(self):
        for name, p in self.named_parameters():
            if not p.requires_grad:
                print(f"🧊 Frozen parameter: {name} -- {p.size()}", flush=True)
            else:
                print(f"🔥 Tuned parameter: {name} -- {p.size()}", flush=True)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def show_n_params(self, parameters, return_str=True):
        tot = 0
        for p in parameters:
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e9:
                return "{:.1f}B".format(tot / 1e9)
            elif tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def get_opt_params(self):
        # uniform optimization
        params_group = []
        for p in self.parameters():
            if p.requires_grad:
                params_group.append(p)

        return params_group
    
    def image_preprocessor(self, images):
        """
        Process input images
        output:
            images -> pixel_values: (N, 3, W, H) 
        """
        images = images.to(self.device)
        return images
    
    def processor(
            self, 
            text: Union[TextInput, List[TextInput]] = None,  # (B, )
            images = None,  # batch of image list;  different number of images per batch
            objects = None, # batch of image list;  different number of images per batch 
            ):
        """
        Process input text
        output:
            text -> inputs['input_ids']: (B, T) token indexes of characters T: max length of input text
                    input['attention_mask']: (B, T) mask of input text
        """
        inputs = {}

        # text processing
        if text is None:
            inputs['input_ids'] = None
            inputs['attention_mask'] = None
        else:
            self.llm_tokenizer.padding_side = 'left'
            inputs['input_ids'] = self.llm_tokenizer(text, return_tensors='pt', padding='longest').input_ids
            inputs['attention_mask'] = self.llm_tokenizer(text, return_tensors='pt', padding='longest').attention_mask
        
        return inputs

    def build_embeds(self,  
                            img_mask = None,        # (B, N_max_img)
                            scene_dict = None,      # 
                            input_ids = None,       # (B, T)
                            attention_mask = None,  # (B, T)    
                            img_sp_token = 30861,   # vicuna 
                            scene_sp_token = 31495  # vicuna
                            ): 
        """
        Build input embeddings for LLM
        input:
            keys:
                pixel_values      # (B, N_max_img, 3, W, H)
                obj_values       # (B, N_max_obj, num_points, 6)
                obj_mask         # (B, N_max_obj)
                scene_mask = None,      # (B, N_obj)
        output:
            inputs_embeds: (B, T, D) ;  D: dimension of embeddings
            attention_mask: (B, T)

        """

        input_ids = input_ids.to(self.device)
        inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)

        # build image embeddings  
        if 'msr3d_imgs' in scene_dict:
            img_mask = scene_dict['msr3d_img_masks']
            if img_mask.sum(1).sum(0) > 0:
                pixel_values = scene_dict['msr3d_imgs']
                pixel_values = pixel_values[img_mask.bool()]   # (number of images, W, H, 3)
                img_fts = self.image_encoder(pixel_values)
                img_embeds = self.llm_proj_img(img_fts)        # (number of images, image_token_len, D)
                img_embeds_index = torch.where(input_ids == img_sp_token) # (number of images, 1)
                # img_embeds = torch.tensor(img_embeds, dtype=inputs_embeds.dtype)
                # HACK: gradient update
                img_embeds = img_embeds.to(dtype=inputs_embeds.dtype)
                inputs_embeds[img_embeds_index] = img_embeds.reshape(-1, img_embeds.shape[-1]) # (B, image_token_len, D)
        else:
            img_mask = scene_dict['img_masks']
            img_embeds = scene_dict['img_tokens']
            # img_embeds = torch.tensor(img_embeds, dtype=inputs_embeds.dtype)
            img_embeds = img_embeds.to(dtype=inputs_embeds.dtype)
            img_embeds_index = torch.where(input_ids == img_sp_token)
            inputs_embeds[img_embeds_index] = img_embeds.reshape(-1, img_embeds.shape[-1])
            # build su
            situation_input_ids = self.llm_tokenizer(scene_dict['prompt_middle_1'], return_tensors='pt', padding=False).input_ids
            situation_input_ids = situation_input_ids.to(self.device)
            situation_input_ids = situation_input_ids[:, 1:]     # remove bos token
            subsequence_search = situation_input_ids[0]     # extract the sequence for searching
            situation_mask = torch.zeros_like(situation_input_ids, dtype=torch.bool, device=situation_input_ids.device)
            situation_mask = situation_mask.unsqueeze(-1)
            situation_embeds_index = find_subsequence(input_ids, subsequence_search)
            img_mask = img_mask.unsqueeze(-1)
            attention_mask = attention_mask.unsqueeze(-1)
            # attention_mask = torch.tensor(attention_mask, dtype=img_mask.dtype).to(self.device)
            attention_mask = attention_mask.to(dtype=img_mask.dtype).to(self.device)
            # insert img_mask and situation_mask on attention mask
            attention_mask[img_embeds_index] = img_mask.reshape(-1, img_mask.shape[-1])
            attention_mask[situation_embeds_index] = situation_mask.reshape(-1, situation_mask.shape[-1])
            attention_mask = attention_mask.squeeze(-1)
        
        # build scene embeddings
        if scene_dict is not None:
            # scene_values = scene_values[scene_mask.bool()] # TODO, only one scene for each prompt text currently, the code should be compatible with LEO's training pipeline
            # embed()
            if 'obj_tokens' not in scene_dict:
                scene_dict = self.visual_prompter(scene_dict)

            scene_embeds = self.llm_proj(scene_dict['obj_tokens'].to(self.device))

            scene_embeds = scene_embeds.to(dtype=inputs_embeds.dtype)
            scene_embeds_index = torch.where(input_ids == scene_sp_token)
            scene_mask = scene_dict['obj_masks'].to(self.device)
            inputs_embeds[scene_embeds_index] = scene_embeds.reshape(-1, scene_embeds.shape[-1])
            scene_mask = scene_mask.unsqueeze(-1)
            attention_mask = attention_mask.unsqueeze(-1)
            attention_mask = attention_mask.to(dtype=scene_mask.dtype).to(self.device)
            attention_mask[scene_embeds_index] = scene_mask.reshape(-1, scene_mask.shape[-1])
            attention_mask = attention_mask.squeeze(-1)
          
        return inputs_embeds, attention_mask
    
    def build_text_prompt(self, data_dict):
        """
        Build text prompt for LLM
        required input dict key: 
            role_prompt = "You are an AI visual assistant situated in a 3D scene. "\
                "You can perceive (1) an ego-view image (accessible when necessary) and (2) the objects (including yourself) in the scene (always accessible). "\
                "You should properly respond to the USER's instruction according to the given visual information. "
            situation_prompt = "{situation}"
            egoview_prompt = "Ego-view image:"
            objects_prompt = "Objects (including you) in the scene:"
            task_prompt = "USER: {instruction} ASSISTANT:"
        output:
            data_dict['prompt']: (B, T) ;  T: max length of input text
        """
        scene_placeholder = "景"
        scene_replace_holder = "".join([scene_placeholder]*self.scene_token_len)
        image_placeholder = "图"
        image_replace_holder = "".join([image_placeholder])     # TODO(lhxk): if using avg pooling, then an image occupies 1 tokon, if no avg pooling, then an image occupies more tokens
        
        if 'msr3d_prompt' not in data_dict:
            # insert replace holder for LEO data
            bs = len(data_dict['prompt_before_obj'])
            scene_replace_holder_list = [f" {scene_replace_holder}"]*bs
            image_replace_holder_list = [f" {image_replace_holder}"]*bs

            data_dict['prompt'] = [f'{prompt_before_obj} {prompt_middle_1}{image_replace_holder}. {prompt_middle_2}{scene_replace_holder}. {prompt_after_obj}' for prompt_before_obj, prompt_middle_1, image_replace_holder, prompt_middle_2, scene_replace_holder, prompt_after_obj in zip(data_dict['prompt_before_obj'], data_dict['prompt_middle_1'], image_replace_holder_list, data_dict['prompt_middle_2'], scene_replace_holder_list, data_dict['prompt_after_obj'])]

        else:
            bs = len(data_dict['msr3d_prompt'])
            for k, v in data_dict.items():
                if k == "msr3d_prompt":
                    for i in range(len(data_dict["msr3d_prompt"])):
                        data_dict["msr3d_prompt"][i] = data_dict["msr3d_prompt"][i].replace(scene_placeholder, scene_replace_holder).replace(image_placeholder, image_replace_holder)
            data_dict['prompt'] = data_dict["msr3d_prompt"]

        return data_dict

    def forward(self, data_dict):
        # Input:
        #   required keys:
        #       prompt_before_obj: list of str, (B,)
        #       prompt_middle_1: list of str, (B,)
        #       prompt_middle_2: list of str, (B,)
        #       prompt_after_obj: list of str, (B,)
        #       obj_fts: (B, N, C)
        #       obj_masks: (B, N) 1 -- not masked, 0 -- masked
        #       obj_locs: (B, N, 6)
        #       img_fts: (B, 3, W, H) -- only 1 image per seq at this point
        #       img_masks: (B, 1) 1 -- not masked, 0 -- masked
        #       text_output: list of str, (B,)
        #       <del>past_action_output: (B, cfg.action_transformer.history_length)</del>
        #       <del>action_output: (B, cfg.action_transformer.num_pred)</del>
        #       -when ose3d.use_anchor and ose3d.use_orientation:
        #           anchor_orientation: (B, C)
        #           anchor_locs: (B, 3)

        assert 'action_output' not in data_dict, "action transformer is deprecated."

        with torch.no_grad():
            with maybe_autocast(self):
                img_fts = self.image_encoder(data_dict['img_fts'])
        data_dict['img_tokens'] = self.llm_proj_img(img_fts.requires_grad_())

        data_dict = self.build_text_prompt(data_dict)

        inputs = self.processor(text=data_dict["prompt"])

        # try:
        inputs_embeds, attention_mask = self.build_embeds(scene_dict=data_dict, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        # (B, T1+O+T2, D), (B, T1+O+T2)
        # except:
        #     ValueError("Error in building embeddings")
        #     print(f"source: {data_dict['source']}")
        #     print(f"scan id: {data_dict['scan_id']}")
        #     print(f"index: {data_dict['index']}")
        #     print(data_dict['prompt'])
        bs = inputs_embeds.shape[0]
        input_length = inputs_embeds.size(1) # T1
            
        self.llm_tokenizer.padding_side = 'right'
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in data_dict['text_output']],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_out_len,
        ).to(self.device)


        text_output_embeds = self.llm_model.get_input_embeddings()(text_output_tokens.input_ids)   # (B, T3, D)
        inputs_embeds = torch.cat([inputs_embeds, text_output_embeds], dim=1)   # (B, T1+O+T2+T3, D)
        attention_mask = torch.cat([attention_mask, text_output_tokens.attention_mask], dim=1)   # (B, T1+O+T2+T3)
        # construct targets
        targets = torch.zeros_like(attention_mask).long().fill_(-100)   # (B, T1+O+T2+T3)

        # only apply loss to answer tokens
        targets_idx = text_output_tokens.attention_mask.bool()
        targets[:, -targets_idx.shape[1]:][targets_idx] = text_output_tokens.input_ids[targets_idx]

        # do not predict bos token, regard it as condition instead
        targets[:, -targets_idx.shape[1]] = -100

        if 'action_output' in data_dict:
            if 'past_action_output' not in data_dict:
                # TODO(jxma): simply put [STOP] action into the history. Here we assume 0 indicates [STOP]
                act_input_embeds = self.action_embedding(torch.zeros(bs, self.action_history_length).to(self.device).long())
            else:
                act_input_embeds = self.action_embedding(data_dict['past_action_output'].to(self.device).long())
            if self.action_use_2d:
                act_input_embeds = torch.cat([data_dict['img_tokens'], act_input_embeds], dim=1)
            act_input_length = act_input_embeds.size(1)
            act_input_embeds = torch.cat([act_input_embeds, self.action_query.expand(bs, -1, -1)], dim=1)

        # Final context truncation
        # inputs_embeds = inputs_embeds[:, -self.max_context_len:]
        # attention_mask = attention_mask[:, -self.max_context_len:]
        # targets = targets[:, -self.max_context_len:]
        with maybe_autocast(self):
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
            if 'action_output' in data_dict:
                # Note: we ensure only the output tokens corresponding to input
                # tokens (query tokens excluded) are used by action transformer.
                outputs_act = self.action_transformer(
                    inputs_embeds=act_input_embeds,
                    encoder_hidden_states=outputs.hidden_states[-1][:, :input_length],
                    return_dict=True,
                    output_hidden_states=True,
                )

        logits = outputs.logits.float()

        # different from the loss inside `llm_model.forward`, here we take mean of each sequence instead of sum
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        num_tokens_for_loss = (shift_labels >= 0).int().sum(1)   # (B,)

        shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
        shift_labels = rearrange(shift_labels, 'b t -> (b t)')

        shift_labels = shift_labels.to(shift_logits.device)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        loss = rearrange(loss, '(b t) -> b t', b=bs)
        loss = loss.sum(1) / num_tokens_for_loss   # (B,)

        data_dict.update({'loss': loss})

        return data_dict

    @torch.no_grad()
    def generate(
            self,
            data_dict,
            use_nucleus_sampling=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=3.0,
            length_penalty=1,
            num_captions=1,
            temperature=1,
            pred_action=False,
    ):
        # data_dict:
        #   required keys:
        #       obj_fts: (B, N, C)
        #       obj_masks: (B, N) 1 -- not masked, 0 -- masked
        #       obj_locs: (B, N, 6)
        #       img_fts: (B, 3, W, H) -- only 1 image per seq at this point
        #       img_masks: (B, 1) 1 -- not masked, 0 -- masked
        #       <del>past_action_output: (B, cfg.action_transformer.history_length)</del>
        #       -when ose3d.use_anchor and ose3d.use_orientation:
        #           anchor_orientation: (B, C)
        #           anchor_locs: (B, 3)
        
        data_dict = self.build_text_prompt(data_dict)

        inputs = self.processor(text=data_dict["prompt"])
        data_dict['img_tokens'] = self.llm_proj_img(self.image_encoder(data_dict['img_fts']))
        try:
            inputs_embeds, attention_mask = self.build_embeds(scene_dict=data_dict, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            bs = inputs_embeds.shape[0]
        except:
            ValueError("Error in building embeddings")
            print(f"source: {data_dict['source']}")
            print(f"scan id: {data_dict['scan_id']}")
            print(f"index: {data_dict['index']}")
            print(data_dict['prompt'])


        # give bos token as condition
        bos_tokens = self.llm_tokenizer(
            [self.llm_tokenizer.bos_token] * bs,
            return_tensors='pt',
        ).to(self.device)
        bos_tokens_ids = bos_tokens.input_ids[:, 0:1]   # (B, 1)
        bos_tokens_attn = bos_tokens.attention_mask[:, 0:1]   # (B, 1)

        # prepare a `bos_token`
        bos_embeds = self.llm_model.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)
        inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
        attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)

        with maybe_autocast(self):
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2   # convert output id 0 (unk_token) to 2 (eos_token)

        # pad to same length for accelerator to gather
        samples = data_dict
        samples['output_tokens'] = pad_tensors(outputs, dim=1, lens=self.max_out_len, pad=2)

        if pred_action:
            samples = self.predict_action(samples)

        return samples

    @torch.no_grad()
    def predict_answers(self, samples, answer_list, num_ans_candidates=128):
        """
        (1) Generate the first token of answers using decoder and select `num_ans_candidates` most probable ones.
        (2) Then select answers from answer list, which start with the probable tokens.
        (3) Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.
        """
        # Input:
        #   required keys:
        #       prompt_before_obj: (B,)
        #       prompt_after_obj: (B,)
        #       obj_fts: (B, N, C)
        #       obj_masks: (B, N) 1 -- not masked, 0 -- masked
        #       obj_locs: (B, N, 6)
        #       img_fts: (B, 3, W, H) -- only 1 image per seq at this point
        #       img_masks: (B, 1) 1 -- not masked, 0 -- masked
        #       -when ose3d.use_anchor and ose3d.use_orientation:
        #           anchor_orientation: (B, C)
        #           anchor_locs: (B, 3)
        #
        num_ans_candidates = min(num_ans_candidates, len(answer_list))

        self.llm_tokenizer.padding_side = 'right'
        answer_candidates = self.llm_tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(self.device)

        answer_ids = answer_candidates.input_ids
        answer_atts = answer_candidates.attention_mask

        # (1)
        if 'obj_tokens' not in samples:
            samples = self.visual_prompter(samples)

        samples['obj_tokens'] = self.llm_proj(samples['obj_tokens'].to(self.device))
        # samples['obj_tokens'] = samples['obj_tokens'] + self.obj_type_embed

        samples['img_tokens'] = self.llm_proj_img(self.image_encoder(samples['img_fts']))

        inputs_embeds, attention_mask = self.build_right_justified_sequence(data_dict=samples)
        bs = inputs_embeds.shape[0]

        # give bos token as condition
        bos_tokens_ids = answer_ids[0, 0].view(1, 1).repeat(bs, 1)   # (B, 1)
        bos_tokens_attn = answer_atts[0, 0].view(1, 1).repeat(bs, 1)   # (B, 1)

        bos_embeds = self.llm_model.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)
        inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
        attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)

        with maybe_autocast(self):
            start_output = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
        logits = start_output.logits[:, -1, :]   # first predicted token's logit

        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)
        # (bs, num_ans_candidates)

        # (2)
        ans_ids = []
        ans_atts = []
        for topk_id in topk_ids:
            ans_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            ans_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        ans_ids = torch.cat(ans_ids, dim=0)
        ans_atts = torch.cat(ans_atts, dim=0)
        # (B * num_ans_candidates, T3)

        inputs_embeds = inputs_embeds.repeat_interleave(num_ans_candidates, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_ans_candidates, dim=0)
        # (B * num_ans_candidates, T1+O+T2+1, D), (B * num_ans_candidates, T1+O+T2+1)

        # truncate the appended bos token before concat
        inputs_embeds = inputs_embeds[:, :-1, :]
        attention_mask = attention_mask[:, :-1]
        # (B * num_ans_candidates, T1+O+T2, D), (B * num_ans_candidates, T1+O+T2)

        ans_embeds = self.llm_model.get_input_embeddings()(ans_ids)   # (B * num_ans_candidates, T3, D)
        inputs_embeds = torch.cat([inputs_embeds, ans_embeds], dim=1)   # (B * num_ans_candidates, T1+O+T2+T3, D)
        attention_mask = torch.cat([attention_mask, ans_atts], dim=1)   # (B * num_ans_candidates, T1+O+T2+T3)

        targets_ids = torch.zeros_like(attention_mask).long().fill_(-100)   # (B * num_ans_candidates, T1+O+T2+T3)
        # only apply loss to answer tokens
        targets_idx = ans_atts.bool()
        targets_ids[:, -targets_idx.shape[1]:][targets_idx] = ans_ids[targets_idx]

        # ignore the prediction of bos token
        targets_ids[:, -targets_idx.shape[1]] = -100

        # (3)
        with maybe_autocast(self):
            output = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=targets_ids,
                return_dict=True
            )

        logits = output.logits.float()

        # get loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets_ids[..., 1:].contiguous()
        num_tokens_for_loss = (shift_labels >= 0).int().sum(1)

        shift_logits = rearrange(shift_logits, 'b t v -> (b t) v')
        shift_labels = rearrange(shift_labels, 'b t -> (b t)')

        shift_labels = shift_labels.to(shift_logits.device)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')   # get loss per token

        loss = rearrange(loss, '(b t) -> b t', b = bs * num_ans_candidates)
        loss = loss.sum(1) / num_tokens_for_loss   # get loss per sequence, average over tokens
        loss = rearrange(loss, '(b1 b2) -> b1 b2', b1=bs)

        max_topk_ids = (-loss).argmax(dim=1)
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids]

        samples['answers_id'] = max_ids
        samples['answers'] = [answer_list[max_id] for max_id in max_ids]

        return samples
