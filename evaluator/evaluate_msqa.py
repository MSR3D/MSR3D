from utils import load_json, execute_chat, EM_Evaluator
import re
import os
import torch
from tqdm import tqdm
from types import SimpleNamespace

def extract_question(text):
    # Using regular expression to find text between 'USER:' and 'ASSISTANT:'
    match = re.search(r"USER: (.*?) ASSISTANT:", text)
    return match.group(1) if match else None

def extract_number(text):
    # Using regular expression to find number in text
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None

class LLMEvaluator():
    def __init__(self, config):
        self.cfg = config
        self.eval_dict = {"total_cnt": 0}
        self.metric_type_list = ['gpt_score', 'em1', 'em1_strict', 'cider', 'bleu', 'meteor', 'rouge']
        for metric_type in self.metric_type_list:
            self.eval_dict[metric_type] = 0
    
    def update(self, score_dict):
        '''
            update the evaluation results
        '''
        for metric_type in score_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] += score_dict[metric_type]
        self.eval_dict["total_cnt"] += 1
    
    def summary(self):
        '''
        '''
        # self.eval_dict["gpt_score"] = self.eval_dict["gpt_score"]/self.eval_dict["total_cnt"]
        for metric_type in self.eval_dict:
            if metric_type in self.metric_type_list:
                self.eval_dict[metric_type] = self.eval_dict[metric_type]/self.eval_dict["total_cnt"]
        return self.eval_dict

    def get_gpt_score(self, question, answer, gt):
        '''
            evaluate the results
        '''
        model = self.cfg.gpt_model
        api_key = self.cfg.api_key
        api_version = self.cfg.api_version
        region = self.cfg.region
        messages = load_json(self.cfg.gpt_score_prompt_path)
        user_prompt = "\n".join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({"role": "user", "content": user_prompt})
        response = execute_chat(messages, api_version, api_key, model, region)
        score = extract_number(response)
        return score

class MSQAEvaluator():
    '''
        process the data from files
    '''
    def __init__(self, config):
        self.cfg = SimpleNamespace(**config)
        self.eval_dict = {"gpt_score": 0, "total_cnt": 0}
        self.evaluator = LLMEvaluator(self.cfg)

    def eval_metrics(self):
        
        dataset_names_list = self.cfg.evaluate_dataset
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        result_dict = load_json(self.cfg.result_file)
        result_scores_dict = {}
        for dataset_name in dataset_names_list:
            result_dict_list = result_dict[dataset_name]
            score_list = []
            for i in tqdm(range(len(result_dict_list))):
                result_dict = result_dict_list[i]
                if 'question' in result_dict:
                    question = result_dict["question"]
                else:
                    if "instruction" in result_dict:
                        question = extract_question(result_dict["instruction"])  
                gt = result_dict["response_gt"][0]
                answer = result_dict["response_pred"]
                index = result_dict["index"]
                scored_dict = {"question": question, "answer": answer, "gt": gt}
                if self.cfg.gpt_score_flag:
                    score = self.evaluator.get_gpt_score(question, answer, gt)
                    scored_dict['gpt_score'] = (score-1)*25
                lang_evaluator = EM_Evaluator()
                scored_dict.update(lang_evaluator.eval_instance(answer, [gt]))
                self.evaluator.update(scored_dict)
                if 'type' in result_dict:
                    scored_dict['type'] = result_dict['type']
                score_list.append(scored_dict)
            result_scores_dict[dataset_name] = score_list
    
        QA_type_list = [
            "counting",
            "existence",
            "attribute",
            "spatial relationship",
            "navigation",
            "refer",
            "affordance",
            "description",
            "room type",
        ]
        statistic_dict = {'scannet': {}, 
                         'RScan': {},
                         'ARKitScenes': {}}
        metric_type_list = ['em1', 'em1_strict']
        result_dict = {}
        file_tag = 'with_gpt_score' if self.cfg.gpt_score_flag else 'without_gpt_score'
        if file_tag == 'with_gpt_score':
            metric_type_list.append('gpt_score')

        for dataset_name in dataset_names_list:
            scores_data = result_scores_dict[dataset_name]
            data_instance_cnt = 0
            for data_instance in scores_data:
                if 'type' in data_instance:
                    data_QA_type = data_instance['type']
                else:
                    ValueError("No type in data_instance")
                for metric_type in metric_type_list:
                    if metric_type not in statistic_dict[dataset_name]:
                        statistic_dict[dataset_name][metric_type] = {}
                    for QA_type in QA_type_list:
                        if QA_type in data_QA_type.lower():
                            if QA_type not in statistic_dict[dataset_name][metric_type]:
                                statistic_dict[dataset_name][metric_type][QA_type] = {'score': [], 'cnt': 0, "avg": 0}
                            statistic_dict[dataset_name][metric_type][QA_type]['score'].append(data_instance[metric_type])
                            statistic_dict[dataset_name][metric_type][QA_type]['cnt'] += 1

        for dataset_name in dataset_names_list:
            for metric_type in metric_type_list:
                for QA_type in QA_type_list:
                    if QA_type in statistic_dict[dataset_name][metric_type]:
                        statistic_dict[dataset_name][metric_type][QA_type]['avg'] = sum(statistic_dict[dataset_name][metric_type][QA_type]['score'])/statistic_dict[dataset_name][metric_type][QA_type]['cnt']
        
        for metric_type in metric_type_list:
            statistic_dict[metric_type] = {'overall': {}}
            for QA_type in QA_type_list:
                score_list = []
                cnt = 0
                for dataset_name in dataset_names_list:
                    if QA_type in statistic_dict[dataset_name][metric_type]:
                        score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                        cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                if cnt > 0:
                    statistic_dict[metric_type]['overall'][QA_type] = sum(score_list)/cnt

        merged_QA_type_list = ['counting', 'existence', 'attribute_description', 'spatial_refer', 'navigation', 'others']
        for metric_type in metric_type_list:
            statistic_dict[metric_type]['merged'] = {}

        for metric_type in metric_type_list:
            for QA_type in merged_QA_type_list:
                score_list = []
                cnt = 0
                for dataset_name in dataset_names_list:
                    if QA_type in ['counting', 'existence', 'navigation']:
                        if QA_type in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type][QA_type]['avg'] * statistic_dict[dataset_name][metric_type][QA_type]['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type][QA_type]['cnt']
                    elif QA_type == 'attribute_description':
                        if 'attribute' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['attribute']['avg'] * statistic_dict[dataset_name][metric_type]['attribute']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['attribute']['cnt']
                        if 'description' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['description']['avg'] * statistic_dict[dataset_name][metric_type]['description']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['description']['cnt']
                    elif QA_type == 'spatial_refer':
                        if 'spatial relationship' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['spatial relationship']['avg'] * statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['spatial relationship']['cnt']
                        if 'refer' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['refer']['avg'] * statistic_dict[dataset_name][metric_type]['refer']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['refer']['cnt']
                    elif QA_type == 'others':
                        if 'affordance' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['affordance']['avg'] * statistic_dict[dataset_name][metric_type]['affordance']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['affordance']['cnt']
                        if 'room type' in statistic_dict[dataset_name][metric_type]:
                            score_list.append(statistic_dict[dataset_name][metric_type]['room type']['avg'] * statistic_dict[dataset_name][metric_type]['room type']['cnt'])
                            cnt += statistic_dict[dataset_name][metric_type]['room type']['cnt']
                    else:
                        ValueError("Invalid QA type")
                if cnt > 0:
                    statistic_dict[metric_type]['merged'][QA_type] = sum(score_list)/cnt
                    statistic_dict[metric_type]['merged'][QA_type + '_cnt'] = cnt
            statistic_dict[metric_type]['merged']['weighted_avg_score'] = sum([statistic_dict[metric_type]['merged'][QA_type] * statistic_dict[metric_type]['merged'][QA_type + '_cnt'] for QA_type in merged_QA_type_list])/sum([statistic_dict[metric_type]['merged'][QA_type + '_cnt'] for QA_type in merged_QA_type_list])
            result_dict = {}
            for key in statistic_dict['em1']['merged']:
                if 'cnt' in key:
                    continue
                if 'weighted' in key:
                    result_dict['EM-R_overall'] = statistic_dict['em1']['merged'][key]
                else:
                    result_dict[f'EM-R_{key}'] = statistic_dict['em1']['merged'][key]
            if 'gpt_score' in statistic_dict:
                for key in statistic_dict['gpt_score']['merged']:
                    if 'cnt' in key:
                        continue
                    if 'weighted' in key:
                        result_dict['GPT-Score_overall'] = statistic_dict['gpt_score']['merged'][key]
                    else:
                        result_dict[f'GPT-Score_{key}'] = statistic_dict['gpt_score']['merged'][key]
    
        return result_dict