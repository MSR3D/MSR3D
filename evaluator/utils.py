import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from openai import AzureOpenAI
import json
import os
import numpy as np
import re

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


class EM_Evaluator():
    def __init__(self):

        self.total_count = 0
        self.best_result = -np.inf

    def init_corpus(self):
        self.gt_sentence_mp = []
        self.pred_sentence_mp = []
    
    def answer_match(self, pred, gts):
        for gt in gts:
            if pred == gt:
                return True
            elif ''.join(pred.split()) in ''.join(gt.split()):
                return True
            elif ''.join(gt.split()) in ''.join(pred.split()):
                return True
        return False

    def answer_match_strict(self, pred, gts):
        for gt in gts:
            if pred == gt:
                return True
        return False
    
    def eval_instance(self, answer_pred, answer_gts):

        results_dict = {}
        ## EM@1
        correct1 = 0
        answer_pred = clean_answer(answer_pred)
        answer_gts = [clean_answer(a) for a in answer_gts]
        if self.answer_match(pred=answer_pred, gts=answer_gts):
            correct1 += 1
        
        ## EM strict
        correct_strict = 0
        answer_pred = clean_answer(answer_pred)
        answer_gts = [clean_answer(a) for a in answer_gts]
        if self.answer_match_strict(pred=answer_pred, gts=answer_gts):
            correct_strict += 1

        results_dict['em1'] = correct1
        results_dict['em1_strict'] = correct_strict

        return results_dict

def calculate_average_iou(result_list):
    iou_sums = defaultdict(float)
    iou_counts = defaultdict(int)
    iou_25_sums = defaultdict(float)
    iou_50_sums = defaultdict(float)
    for item in result_list:
        label = item['label'] # Get the label from the dictionary
        iou = item["iou"]
       
        iou_sums[label] += iou
        iou_counts[label] += 1
        if iou > 0.25:
            iou_25_sums[label] += 1
        if iou > 0.50:
            iou_50_sums[label] += 1

    # Calculate average IoU for each label
    average_ious = {label: iou_sums[label] / iou_counts[label] for label in iou_sums}
    average_ious_25 = {label: iou_25_sums[label] / iou_counts[label] for label in iou_25_sums}
    average_ious_50 = {label: iou_50_sums[label] / iou_counts[label] for label in iou_50_sums}

    return average_ious, average_ious_25, average_ious_50

def plot_top_average_iou_and_counts(average_ious, source_refer_data, top_n=20, name="Average IoU"):
    # Sort the labels by average IoU in descending order and select top top_n
    sorted_items = sorted(average_ious.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, avg_ious = zip(*sorted_items)
    training_counts = defaultdict(int)
    all_item = 0
    for item in source_refer_data:
        label = item['object_name']
        training_counts[label] += 1
        all_item += 1
    counts = [training_counts[label] for label in labels]

    print(f'counts: {counts}')
    print(f'all item count in scanrefer: {all_item}')
   
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot average IoU
    ax1.barh(labels, avg_ious, color='skyblue', label=f'{name}')
    ax1.set_xlabel(f'{name}')
    ax1.set_title(f'Top {top_n} Objects by {name} and Object Count')
    ax1.invert_yaxis()  # To display the highest IoU at the top

    # Create a secondary y-axis to plot the object counts
    ax2 = ax1.twiny()
    ax2.barh(labels, counts, color='orange', alpha=0.5, label='Object Count')
    ax2.set_xlabel('Object Count')

    # Add legends
    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')

    plt.show()

def load_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file

def save_to_json(path, json_file):
    with open(path, 'w') as f:
        json.dump(json_file, f)

def execute_chat(messages, api_key, model="gpt-3.5-turbo", region = "canadaeast", temperature=0.11, max_tokens=1024):
    # REGION = "canadaeast"
    # MODEL = "gpt-35-turbo-0125"
    # API_KEY = "<your-key>"

    API_BASE = ""   # NOTE: This should be set to your Azure OpenAI endpoint, e.g., "https://<your-resource-name>.openai.azure.com"
    ENDPOINT = f"{API_BASE}/{region}"


    client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        # print(response.model_dump_json(indent=2))
        # print(response.choices[0].message.content)
        # time.sleep(0.5)
        return response.choices[0].message.content if response.choices[0].message.content else ""
    except Exception as e:
        # print(e)
        return ""
    
def path_verify(path):
    if not os.path.exists(path):
        os.makedirs(path)