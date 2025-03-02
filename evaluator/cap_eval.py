import collections
import json
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim

from evaluator.build import EVALUATOR_REGISTRY
from evaluator.capeval.cider.cider import Cider
from evaluator.capeval.bleu.bleu import Bleu
from evaluator.capeval.meteor.meteor import Meteor
from evaluator.capeval.rouge.rouge import Rouge


@EVALUATOR_REGISTRY.register()
class GenerationEval():
    def __init__(self, cfg, task_name):
        self.eval_dict = {
            'target_metric': [], 'sentence_sim': [],
        }

        self.cider_scorer = Cider()
        self.bleu_scorer = Bleu()
        self.meteor_scorer = Meteor()
        self.rouge_scorer = Rouge()

        self.total_count = 0
        self.best_result = -np.inf

        self.save = cfg.eval.save
        if self.save:
            self.eval_results = []
            self.save_dir = Path(cfg.exp_dir) / 'eval_results' / task_name
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.task_name = task_name
        self.corpus_path = cfg.data.scan2cap.args.corpus if self.task_name.lower() == 'scan2cap' else None
        self.init_corpus()
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def init_corpus(self):
        if self.task_name.lower() == 'scan2cap':
            self.gt_sentence_mp = torch.load(self.corpus_path)
            self.pred_sentence_mp = {}
        else:
            self.gt_sentence_mp = []
            self.pred_sentence_mp = []

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        self.total_count += metrics["total_count"]

        if self.save:
            for i in range(metrics["total_count"]):
                self.eval_results.append({
                    # vision
                    "source": data_dict['source'][i],
                    "scan_id": data_dict['scan_id'][i],
                    "anchor": data_dict['anchor_locs'][i],
                    "iou_flag": data_dict['iou_flag'][i] if 'iou_flag' in data_dict else True,
                    # language
                    "instruction": data_dict['prompt_after_obj'][i],
                    "response_gt": data_dict['text_output'][i],
                    "response_pred": data_dict['output_text'][i],
                })

        for key in self.eval_dict.keys():
            self.eval_dict[key].append(float(metrics[key]) * metrics["total_count"])

    def batch_metrics(self, data_dict):
        metrics = {}
        text_pred = data_dict['output_text']
        text_gt = data_dict['text_output']
        if 'iou_flag' in data_dict:
            iou_flags = data_dict['iou_flag']
        else:
            iou_flags = [True] * len(text_gt)
        metrics['total_count'] = len(text_gt)

        if self.task_name.lower() == 'scan2cap':
            for i in range(len(text_gt)):
                corpus_key = data_dict['corpus_key'][i]
                if iou_flags[i]:
                    self.pred_sentence_mp[corpus_key] = [('sos ' + text_pred[i] + ' eos').replace('. ', ' . ')]
                else:
                    self.pred_sentence_mp[corpus_key] = ["sos eos"]
                    text_pred[i] = ""
        else:
            for i in range(len(text_gt)):
                if iou_flags[i]:
                    self.pred_sentence_mp.append([text_pred[i]])
                else:
                    self.pred_sentence_mp.append([""])
                    text_pred[i] = ""
                self.gt_sentence_mp.append([text_gt[i]])

        # compute sentence similarity
        embed_pred = self.sentence_model.encode(text_pred, convert_to_tensor=True)
        embed_gt = self.sentence_model.encode(text_gt, convert_to_tensor=True)
        sims = pytorch_cos_sim(embed_pred, embed_gt).diag()
        assert len(sims) == metrics['total_count']

        metrics['sentence_sim'] = sims.mean().item()
        metrics['target_metric'] = metrics['sentence_sim']
        return metrics

    def reset(self):
        self.eval_dict = {
            'target_metric': [], 'sentence_sim': [],
        }
        self.total_count = 0
        self.init_corpus()
        if self.save:
            self.eval_results = []

    def record(self, split='val'):
        # record
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count

        if self.task_name.lower() == 'scan2cap':
            # align, accommodate partial evaluation
            self.gt_sentence_mp = {k: self.gt_sentence_mp[k] for k in self.pred_sentence_mp.keys()}
        else:
            self.gt_sentence_mp = {k: v for k, v in enumerate(self.gt_sentence_mp)}
            self.pred_sentence_mp = {k: v for k, v in enumerate(self.pred_sentence_mp)}

        self.eval_dict['cider'] = self.cider_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['bleu'] = self.bleu_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0][-1]
        self.eval_dict['meteor'] = self.meteor_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]
        self.eval_dict['rouge'] = self.rouge_scorer.compute_score(self.gt_sentence_mp, self.pred_sentence_mp)[0]

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False

        if self.save and (is_best or split == 'test'):
            torch.save(self.eval_results, str(self.save_dir / 'results.pt'))

        return is_best, self.eval_dict
