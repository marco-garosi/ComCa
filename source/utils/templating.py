from typing import Union

import json
import os


def get_template(base_dir: str, id: Union[int, str]):
    try:
        with open(os.path.join(base_dir, 'llm', 'template.txt'), 'r') as f:
            template = f.read()
    except:
        template = None

    try:
        with open(os.path.join(base_dir, 'llm', str(id), 'system-prompt.txt'), 'r') as f:
            system_prompt = f.read()
    except:
        system_prompt = None

    try:
        with open(os.path.join(base_dir, 'llm', str(id), 'user-message.txt'), 'r') as f:
            user_message = f.read()
    except:
        user_message = None

    return system_prompt, user_message


def load_templates_cache_generation(templates_file: str, base_dir: str = ''):
    with open(os.path.join(base_dir, 'res', 'templates', 'sd', templates_file), 'r') as f:
        return json.load(f)
    

def load_template_vqa_ovad(template_file: str, model_slug: str, base_dir: str =''):
    with open(os.path.join(base_dir, 'res', 'templates', 'vqa', model_slug, template_file), 'r') as f:
        return f.read()
