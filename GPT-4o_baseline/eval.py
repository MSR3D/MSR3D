import os
import os.path as osp
import json
import re
import numpy as np
import argparse
from tools.gpt_utils import GPTAssistant


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-based situated QA evaluation.")
    parser.add_argument('--dataset', type=str, default='scannet', help='Dataset name (e.g., scannet)')
    parser.add_argument('--engine', type=str, default='gpt-4o-2024-08-06', help='LLM engine name')
    parser.add_argument('--debug', action='store_true', help='Debug mode (only run one example)')
    parser.add_argument('--root_path', type=str, default='',
                        help='Root path to the repo')
    parser.add_argument('--data_base_dir', type=str,
                        default='',
                        help='Base directory for text annotation data')
    parser.add_argument('--img_dir', type=str,
                        default='',
                        help='Directory containing image files')
    return parser.parse_args()


def ensure_working_directory(root_path):
    if root_path not in os.getcwd():
        os.chdir(root_path)


def format_check_func(text: str) -> bool:
    return text is not None


def format_refine_func(response: str) -> str:
    return response.replace('Answer:', '').strip()


def prompt_format_func(input_data: dict) -> str:
    scene_format = (
        "inst_name: [x, y, z], [h, w, d], color, 3D shape, material, usage, texture, structure, state;"
    )
    answer_format = "Answer:"

    input_data["location"] = [round(x, 3) for x in input_data["location"]]

    if "orientation_angle" not in input_data:
        input_data["orientation_angle"] = np.arctan2(
            input_data["orientation"][1], input_data["orientation"][0]
        )
    input_data["orientation_angle"] = round(input_data["orientation_angle"], 3)

    prompt = f"""
You are an AI visual assistant situated in a 3D scene. 
You can perceive the objects (including yourself) in the scene. 
The scene representation is given in a dict format such as {scene_format}

All object instances in this room are given, along with their center point position and size. 
The center points are represented by a 3D coordinate (x, y, z) in meters, and the bounding boxes are (h, w, d).

The objects in the scene are: {input_data['scene_info_str']}

You are located at {input_data['location']} and facing direction in x-y plane with angle {input_data['orientation_angle']}.
Your situation is: {input_data['situation']}

USER: {input_data['question']}

You should respond according to the given information. The answer should follow this format:
{answer_format}

There are some objects represented by images. The image order is: {input_data['img_order']}

ASSISTANT:"""
    return prompt.strip()


def load_annotations(data_base_dir, dataset):
    file = osp.join(data_base_dir, dataset, f"msqa_{dataset}_test.json")
    with open(file, "r") as f:
        return json.load(f)


def load_scan_and_attr_info(data_base_dir, dataset):
    scan_info_file = osp.join(data_base_dir, dataset, f"{dataset}_inst_info.json")
    attr_info_file = osp.join(data_base_dir, dataset, f"{dataset}_attribute.json")
    with open(scan_info_file, "r") as f:
        scan_info = json.load(f)
    with open(attr_info_file, "r") as f:
        attr_info = json.load(f)
    return scan_info, attr_info


def construct_scene_str(scan_info_one_scan, attr_info_one_scan):
    for inst_id in scan_info_one_scan:
        if inst_id in attr_info_one_scan:
            scan_info_one_scan[inst_id]['attribute'] = attr_info_one_scan[inst_id]
        else:
            inst_name = scan_info_one_scan[inst_id]['inst_name']
            new_key = f"{inst_name}-{inst_id}"
            scan_info_one_scan[inst_id]['attribute'] = attr_info_one_scan.get(new_key, {})
    scene_info_str = ""
    for inst_id, inst_info in scan_info_one_scan.items():
        attr_str = ", ".join(inst_info['attribute'].values())
        scene_info_str += f"{inst_info['inst_name']}: {inst_info['inst_cent']}, {inst_info['inst_bbox']}, {attr_str}; "
    return scene_info_str


def extract_image_paths(one_qa, scan_id, img_dir):
    pattern = r"<(.*?)>"
    matches = re.findall(pattern, one_qa["situation"]) + re.findall(pattern, one_qa["question"])
    valid_imgs, to_remove = [], []

    for tmp in matches:
        parts = tmp.split('-')
        if len(parts) != 3:
            continue
        inst_label, inst_id, _ = parts
        img_path = osp.join(img_dir, f"{scan_id}_inst{inst_id}_{inst_label}_0.jpg")
        if osp.exists(img_path):
            valid_imgs.append(img_path)
        else:
            to_remove.append(tmp)
    for invalid in to_remove:
        matches.remove(invalid)
    return str(matches)


def evaluate(args):
    ensure_working_directory(args.root_path)

    dataset = args.dataset
    print(f"[INFO] Evaluating dataset: {dataset}")
    gpt_assistant = GPTAssistant(model_type=args.engine, cache_dir=f"./cache/{dataset}_{args.engine}")

    data_all = load_annotations(args.data_base_dir, dataset)
    scan_info, attr_info = load_scan_and_attr_info(args.data_base_dir, dataset)
    output_dict = {}
    failed_list = []
    total_cost = 0

    for one_qa in data_all:
        scan_id = one_qa["scan_id"]
        scan_info_scan = scan_info[scan_id]
        attr_info_scan = attr_info[scan_id]

        scene_str = construct_scene_str(scan_info_scan, attr_info_scan)
        one_qa["scene_info"] = scan_info_scan
        one_qa["scene_info_str"] = scene_str
        one_qa["img_order"] = extract_image_paths(one_qa, scan_id, args.img_dir)

        output_suffix = f"{dataset}__{scan_id}__{one_qa['index']}"
        res = gpt_assistant.prompt_one_quest(
            out_suffix=output_suffix,
            text_data=one_qa,
            img_data_list=[],
            prompt_format_func=prompt_format_func,
            format_check_func=format_check_func,
            format_refine_func=format_refine_func,
        )

        total_cost += res["price"]
        if not res["pass_format_check"]:
            failed_list.append(output_suffix)
            continue

        output_dict.setdefault(scan_id, {})[one_qa["index"]] = {
            "gpt_response": res["content"],
            "question": one_qa["question"],
            "answer_gt": one_qa["answers"],
            "cost": res["price"],
        }

        if args.debug:
            break

    result_path = f"./results/{dataset}_{args.engine}.json"
    with open(result_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    print(f"[INFO] Finished. Total examples: {len(data_all)}, Failed: {len(failed_list)}, Total Cost: ${total_cost:.4f}")
    if failed_list:
        print("[INFO] Failed list:", failed_list)


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
