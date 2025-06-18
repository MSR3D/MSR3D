import os
import json
import base64
from openai import AzureOpenAI

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class GPTAssistant():
    def __init__(self, model_type = "gpt-35-turbo-0125", cache_dir = None, use_cache = True) -> None:
        REGION_map = {
            "gpt-35-turbo-0125" : "canadaeast",
            "gpt-4-0125-preview" : "eastus",
            "gpt-4-vision-preview" : "japaneast", # "australiaeast",
            "gpt-4-1106-preview" : "australiaeast",
            "gpt-4-0613 Legacy" : "australiaeast",
            "gpt-4o-2024-08-06": "northcentralus"
        }
        
        "canadaeast"
        self.model_type = model_type
        API_KEY = "265101dc671d6a30d5f7bc91da7ea7ec"

        API_BASE = ""   # TODO: set your Azure OpenAI API base URL here
        ENDPOINT = f"{API_BASE}/{REGION_map[self.model_type]}"

        self.client = AzureOpenAI(
            api_key = API_KEY,
            api_version = "2024-02-01",
            azure_endpoint = ENDPOINT,
        )
        self.max_token = 1000

        self.price_dict = {
            "gpt-35-turbo-0125" : [0.0005, 0.0015],
            "gpt-4-0125-preview" : [0.01, 0.03],
            "gpt-4-vision-preview" : [0.01, 0.03],
            "gpt-4-1106-preview" : [0.01, 0.03],
            "gpt-4-0613 Legacy" : [0.03, 0.06],
        }

        self.use_cache = use_cache
        self.cache_dir = cache_dir
        print('use cache', self.use_cache)
        if self.cache_dir is not None and not os.path.exists(self.cache_dir):
            print('making', self.cache_dir)
            os.makedirs(self.cache_dir)

    def prompt_one_quest(self, out_suffix, text_data, img_data_list = [], img_input_quality = 'low',
                         prompt_format_func = None, format_check_func = None, format_refine_func = None):
        output_file_name = os.path.join(self.cache_dir, f"{out_suffix}.json")
        if self.use_cache and os.path.exists(output_file_name):
            with open(output_file_name, 'r') as f:
                res = json.load(f)
            if 'pass_format_check' in res and res['pass_format_check']:
                return res
        
        if prompt_format_func is not None:
            prompt = prompt_format_func(text_data)
        else:
            raise NotImplementedError("prompt function not provided...")
        
        if len(img_data_list) == 0:
            content_info = prompt
        else:
            assert self.model_type.find('vision') != -1, print('please use gpt vision version (such as "gpt-4-vision-preview")')
            content_info =[{
                "type": "text",
                "text": prompt
            }]
            for image_path in img_data_list:
                img_base = encode_image(image_path)
                content_info.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base}",
                        "detail" : img_input_quality,
                    }
                })

        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": content_info}
                ],
                max_tokens = self.max_token
            )
            # print(response)

            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            price = prompt_tokens / 1000 * self.price_dict[self.model_type][0]
            price += output_tokens / 1000 * self.price_dict[self.model_type][1]
        except:
            content = ""
            price = 0
            pass_format_check = False

        if format_check_func is None:
            pass_format_check = True
        else:
            pass_format_check = format_check_func(content)

        if format_refine_func is not None:
            content = format_refine_func(content)

        res = {
            "prompt" : prompt,
            "content" : content,
            "price" : price,
            "pass_format_check" : pass_format_check
        }

        if self.use_cache:
            with open(output_file_name, 'w') as f:
                json.dump(res, f, indent = 4)

        return res