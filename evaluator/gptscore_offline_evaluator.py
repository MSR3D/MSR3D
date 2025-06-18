import random
from evaluate_msqa import MSQAEvaluator

def evaluate(user_submission_file, phase_codename="test", **kwargs):
    print("Starting Evaluation.....")

    configs = {
        "result_file": user_submission_file,
        'gpt_score_flag': False,
        "evaluate_dataset": ['scannet', 'RScan', 'ARKitScenes'],
        "gpt_score_flag": False,
        "gpt_model": "gpt-4o-2024-08-06",
        "api_key": "",
        "api_version": "azure",
        "region": "",
    }
    """
        required structure of json file:
        {
            "scannet": {    ["response_pred",  # string, 
                            "type",           # string
                            "question",       # string
                            "index",          # int
                            ],
                            ...
                            }  
            "RScan": {...}
            "ARKitScenes": {...}  
            }
    """
    
    output = {}
    if phase_codename == "test":
        print("Evaluating for Test Phase")
        msqa_evaluator = MSQAEvaluator(configs)
        output = msqa_evaluator.eval_metrics()
        print("Completed evaluation for Test Phase")
    return output
