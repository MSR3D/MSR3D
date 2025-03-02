import os
import json
import collections
from pathlib import Path

import numpy as np
import torch

from data.data_utils import SQA3DAnswer, clean_answer
from evaluator.build import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register()
class SQA3DEval():
    def __init__(self, cfg, task_name):
        self.eval_dict = {
            'target_metric': [], 'obj_cls_raw_acc': [], 'obj_cls_pre_acc': [],
            'obj_cls_post_acc': [],'ans1_acc': [], 'ans10_acc': [],
            'type0_acc': [], 'type1_acc': [], 'type2_acc': [],
            'type3_acc': [], 'type4_acc': [], 'type5_acc': []
        }
        # run
        self.total_count = 0
        self.type_count = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10, 'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10
        }
        self.best_result = -np.inf
        self.base_dir = cfg.data.scan_family_base

        answer_data = json.load(
            open(os.path.join(self.base_dir, 'annotations/sqa_task/answer_dict.json'), encoding='utf-8')
        )[0]
        answer_counter = []
        for data in answer_data.keys():
            answer_counter.append(data)
        answer_counter = collections.Counter(sorted(answer_counter))
        answer_cands = answer_counter.keys()
        self.answer_vocab = SQA3DAnswer(answer_cands)

        self.save = cfg.eval.save
        if self.save:
            self.eval_results = []
            self.save_dir = Path(cfg.exp_dir) / "eval_results" / task_name
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        batch_count = metrics['total_count']
        self.total_count += batch_count
        for key in metrics:
            if 'type' in key and 'count' in key:
                self.type_count[key] += metrics[key]

        if self.save:
            for i in range(metrics["total_count"]):
                self.eval_results.append({
                    # vision
                    "source": data_dict['source'][i],
                    "scan_id": data_dict['scan_id'][i],
                    "anchor": data_dict['anchor_locs'][i],
                    'anchor_ort': data_dict['anchor_orientation'][i],
                    # language
                    "instruction": data_dict['prompt_after_obj'][i],
                    "response_gt": data_dict['answer_list'][i].split('[answer_seq]'),
                    "response_pred": data_dict['output_text'][i]
                })
        
        # save eval dict
        for key in self.eval_dict.keys():
            if 'type' in key:
                self.eval_dict[key].append(float(metrics[key]) * metrics['type' + key[4] + '_count'])
            else:
                self.eval_dict[key].append(float(metrics[key]) * batch_count)

    def batch_metrics(self, data_dict):
        metrics = {}

        # ans
        choice_1 = data_dict['answer_scores'].argmax(dim=-1)
        choice_10 = torch.topk(data_dict['answer_scores'].detach(), 10, -1)[1]
        correct1 = 0
        correct10 = 0
        correct_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        count_type = {0: 1e-10, 1: 1e-10, 2: 1e-10, 3: 1e-10, 4: 1e-10, 5: 1e-10}
        for i in range(data_dict['answer_label'].shape[0]):
            count_type[data_dict['sqa_type'][i].item()] += 1
            if data_dict['answer_label'][i, choice_1[i]] == 1:
                correct1 += 1
                correct_type[data_dict['sqa_type'][i].item()] += 1
            for j in range(10):
                if data_dict['answer_label'][i, choice_10[i, j]] == 1:
                    correct10 += 1
                    break
        metrics['ans1_acc'] = correct1 / float(len(choice_1))
        metrics['ans10_acc'] = correct10 / float(len(choice_1))
        metrics['answer_top10'] = [
            # TODO: add this answer vocabulary in dataloader
            [self.answer_vocab.itos(choice_10[i, j].item()) for j in range(10)] for i in
            range(choice_10.shape[0])
        ]
        
        # cls, cls_pre
        metrics['obj_cls_post_acc'] = torch.sum(
            torch.argmax(data_dict['obj_cls_post_logits'], dim=2)[data_dict['obj_masks']] ==
            data_dict["obj_labels"][
                data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item())
        metrics['obj_cls_pre_acc'] = torch.sum(
            torch.argmax(data_dict['obj_cls_pre_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][
                data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item())
        metrics['obj_cls_raw_acc'] = torch.sum(
            torch.argmax(data_dict['obj_cls_raw_logits'], dim=2)[data_dict['obj_masks']] == data_dict["obj_labels"][
                data_dict['obj_masks']]).item() / float(data_dict['obj_masks'].sum().item())
        
        # question type acc
        for key in count_type.keys():
            metrics['type' + str(key) + '_acc'] = correct_type[key] / count_type[key]
            metrics['type' + str(key) + '_count'] = count_type[key]

        metrics['target_metric'] = metrics['ans1_acc']
        metrics["total_count"] = data_dict["answer_scores"].shape[0]
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0
        self.type_count = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10, 'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10
        }
        if self.save:
            self.eval_results = []

    def record(self, split='val'):
        # record
        for k, v in self.eval_dict.items():
            if k == "answer_top10":
                continue
            if 'type' in k:
                self.eval_dict[k] = sum(v) / self.type_count['type' + k[4] + '_count']
            else:
                self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict["target_metric"] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False

        if self.save and (is_best or split == 'test'):
            torch.save(self.eval_results, str(self.save_dir / 'results.pt'))

        return is_best, self.eval_dict


@EVALUATOR_REGISTRY.register()
class SQA3DInstructionEval(SQA3DEval):
    def __init__(self, cfg, task_name):
        super().__init__(cfg, task_name)
        self.eval_dict = {
            'target_metric': [], 'ans1_acc_llm': [],
            'type0_acc_llm': [], 'type1_acc_llm': [], 'type2_acc_llm': [],
            'type3_acc_llm': [], 'type4_acc_llm': [], 'type5_acc_llm': []
        }
        self.make_qa_pool()

    def make_qa_pool(self):
        # since `accelerator` cannot gather non-tensor objects, e.g., questions and answers, which are needed for evaluation,
        # we store the questions and answers in a list, and index by the `data_idx` (`question_id`) in `data_dict`
        with open(os.path.join(self.base_dir, 'annotations/sqa_task/balanced/v1_balanced_questions_val_scannetv2.json'), encoding='utf-8') as f:
            sqa3d_val_q = json.load(f)['questions']
        with open(os.path.join(self.base_dir, 'annotations/sqa_task/balanced/v1_balanced_sqa_annotations_val_scannetv2.json'), encoding='utf-8') as f:
            sqa3d_val_a = json.load(f)['annotations']
        with open(os.path.join(self.base_dir, 'annotations/sqa_task/balanced/v1_balanced_questions_test_scannetv2.json'), encoding='utf-8') as f:
            sqa3d_test_q = json.load(f)['questions']
        with open(os.path.join(self.base_dir, 'annotations/sqa_task/balanced/v1_balanced_sqa_annotations_test_scannetv2.json'), encoding='utf-8') as f:
            sqa3d_test_a = json.load(f)['annotations']
        
        self.qa_pool = {}   # { q_id: { 'situations': [<situation>], 'question': <question>, 'answers': [<answer>] } }

        for anno_q in [sqa3d_val_q, sqa3d_test_q]:
            for meta_anno in anno_q:
                self.qa_pool.update({
                    meta_anno['question_id']: {
                        'situations': [meta_anno['situation']] + meta_anno['alternative_situation'],
                        'question': meta_anno['question']
                    }
                })
        for anno_a in [sqa3d_val_a, sqa3d_test_a]:
            for meta_anno in anno_a:
                self.qa_pool[meta_anno['question_id']].update({
                    'answers': [term['answer'] for term in meta_anno['answers'] if term['answer_confidence'] == 'yes']
                })

    def answer_match(self, pred, gts):
        for gt in gts:
            if pred == gt:
                return True
            # elif ''.join(pred.split()) in ''.join(gt.split()):
            #     return True
            # elif ''.join(gt.split()) in ''.join(pred.split()):
            #     return True
        return False

    def batch_metrics(self, data_dict):
        metrics = {
            'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
            'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10,
        }
        
        correct1 = 0
        correct_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        if 'output_text' in data_dict:
            # generation mode
            choice_1 = data_dict['output_text']
            for i in range(len(choice_1)):
                answer_pred = choice_1[i]
                answer_pred = clean_answer(answer_pred)
                q_id = data_dict['data_idx'][i].item()
                answer_gts = self.qa_pool[q_id]['answers']
                answer_gts = [clean_answer(a) for a in answer_gts]
                if self.answer_match(pred=answer_pred, gts=answer_gts):
                    correct1 += 1
                    correct_type[data_dict['sqa_type'][i].item()] += 1
                metrics[f"type{data_dict['sqa_type'][i].item()}_count"] += 1
        else:
            # retrieval mode
            choice_1 = data_dict['answers_id']
            for i in range(len(choice_1)):
                if data_dict['answer_label'][i, choice_1[i]] == 1:
                    correct1 += 1
                    correct_type[data_dict['sqa_type'][i].item()] += 1
                metrics[f"type{data_dict['sqa_type'][i].item()}_count"] += 1
        
        metrics['total_count'] = len(choice_1)
        metrics['ans1_acc_llm'] = correct1 / float(len(choice_1))
        for key in correct_type.keys():
            metrics['type' + str(key) + '_acc_llm'] = correct_type[key] / metrics['type' + str(key) + '_count']

        metrics['target_metric'] = metrics['ans1_acc_llm']
        return metrics
