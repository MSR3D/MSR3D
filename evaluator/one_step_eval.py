from data.datasets.one_step_navi import ONESTEPNAVI_ACTION_SPACE_TOKENIZE
import numpy as np
from evaluator.build import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register()
class ObjNavEval():
    def __init__(self, cfg, task_name):
        self.eval_dict = {
            'target_metric': [], 'accuracy': [],
        }

        self.total_count = 0
        self.best_result = -np.inf

    def update(self, data_dict):
        metrics = self.batch_metrics(data_dict)
        self.total_count += metrics["total_count"]

        for key in self.eval_dict.keys():
            self.eval_dict[key].append(float(metrics[key]) * metrics["total_count"])

    def batch_metrics(self, data_dict):
        metrics = {}
        text_pred = data_dict['output_text']
        text_gt = data_dict['text_output']
        metrics['total_count'] = len(text_gt)

        correct = 0
        for i, j in zip(text_pred, text_gt):
            if i == j:
                correct += 1

        metrics['accuracy'] = correct / len(text_gt)
        metrics['target_metric'] = metrics['accuracy']
        return metrics

    def reset(self):
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        self.total_count = 0

    def record(self):
        # record
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False
        return is_best, self.eval_dict

@EVALUATOR_REGISTRY.register()
class OneStepNavInstructionEval(ObjNavEval):
    def __init__(self, cfg, task_name):
        super().__init__(cfg, task_name)
        self.eval_dict.update({
            'invalid' : []
        })

    def reset(self):
        super().reset()
    
    def batch_metrics(self, data_dict):
        metrics = {}
        text_pred = data_dict['output_text']
        text_gt = data_dict['text_output']
        # print("text", text_pred, text_gt, ONESTEPNAVI_ACTION_SPACE_TOKENIZE.values())

        metrics['total_count'] = len(text_gt)

        correct = 0
        invalid = 0
        for i, j in zip(text_pred, text_gt):
            if i == j:
                correct += 1
            
            if i not in ONESTEPNAVI_ACTION_SPACE_TOKENIZE.values():
                invalid += 1

        metrics['accuracy'] = correct / len(text_gt)
        metrics['invalid'] = invalid / len(text_gt)
        metrics['target_metric'] = metrics['accuracy']
        return metrics

    def record(self, split = None):
        # record
        for k, v in self.eval_dict.items():
            self.eval_dict[k] = sum(v) / self.total_count

        if self.eval_dict['target_metric'] > self.best_result:
            is_best = True
            self.best_result = self.eval_dict['target_metric']
        else:
            is_best = False
        return is_best, self.eval_dict

