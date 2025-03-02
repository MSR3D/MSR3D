import os
import json
import collections
from pathlib import Path

import numpy as np
import torch

from evaluator.cap_eval import GenerationEval
from data.data_utils import clean_answer
from evaluator.build import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register()
class MSQAEval(GenerationEval):
    def __init__(self, cfg, task_name):
        super().__init__(cfg, task_name)
        self.eval_dict = {
            'target_metric': [], 'ans1_acc_llm': []
        }

    def answer_match(self, pred, gts):
        for gt in gts:
            if pred == gt:
                return True
            elif ''.join(pred.split()) in ''.join(gt.split()):
                return True
            elif ''.join(gt.split()) in ''.join(pred.split()):
                return True
        return False

    def batch_metrics(self, data_dict):
        metrics = super().batch_metrics(data_dict)
        correct1 = 0
        for answer_pred, answer_gts in zip(data_dict['output_text'], data_dict['answer_list']):
            answer_pred = clean_answer(answer_pred)
            answer_gts = answer_gts.split('[answer_seq]')
            answer_gts = [clean_answer(a) for a in answer_gts]
            if self.answer_match(pred=answer_pred, gts=answer_gts):
                correct1 += 1

        metrics['total_count'] = len(data_dict['answer_list'])
        metrics['ans1_acc_llm'] = correct1 / float(metrics['total_count'])
        metrics['target_metric'] = metrics['ans1_acc_llm']
        return metrics
    
    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        self.total_count += metrics["total_count"]

        if self.save:
            for i in range(metrics["total_count"]):
                # lhxk: this is a hack to get the instruction
                if "prompt" in data_dict:
                    instruction = data_dict['prompt'][i]
                else:
                    instruction = data_dict['prompt_after_obj'][i]
                self.eval_results.append({
                    # vision
                    "source": data_dict['source'][i],
                    "scan_id": data_dict['scan_id'][i],
                    # language
                    "instruction": instruction,
                    "response_gt": data_dict['answer_list'][i].split('[answer_seq]'),
                    "response_pred": data_dict['output_text'][i],
                    "anchor": data_dict['anchor_locs'][i],
                    "iou_flag": data_dict['iou_flag'][i] if 'iou_flag' in data_dict else True,
                    "index": data_dict['index'][i],
                    "type": data_dict['type'][i],
                })

        for key in self.eval_dict.keys():
            self.eval_dict[key].append(float(metrics[key]) * metrics["total_count"])

    def record(self, split='val'):
        # record
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count

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