from typing import Union

import os
import pickle
import itertools
import torch

from ..utils import clip_encode_text


def embed_prompts(
        att_emb_path,
        annotations,
        use_prompts, object_attribute_templates,
        model,
        object_word: str = '',
        save_result: bool = False,
        allow_loading: bool = True,
        batch_size: int = 1024,
        normalize: bool = True,
        return_prompts: bool = False,
        padding: bool | str = True,
        device: Union[torch.device, str] = 'cpu'
    ):

    # Do attribute embeddings already exist? If so, load them
    if os.path.isfile(att_emb_path) and allow_loading:
        att_emb = pickle.load(open(att_emb_path, 'rb'))
        text_features = torch.Tensor(att_emb['text_features']).to(device)
        len_synonyms = att_emb['len_synonyms']
        all_att_templates = None

    # Otherwise, build prompts using templates and then embed them
    else:
        if isinstance(annotations, dict):
            attributes = annotations.get('attributes', annotations)
        else:
            attributes = annotations
    
        all_att_templates = []

        # Iterate over each attribute
        for att_dict in attributes:
            att_w_type = att_dict['name']
            split = att_w_type.split(':')
            if len(split) == 2:
                att_type, att_list = split[0], split[1]
            elif len(split) == 1:
                att_type = ''
                att_list = split[0]
            is_has = att_dict['is_has_att']
            dobj_name = (
                att_type.replace(' tone', '')
                # So far only for tone worked to remove the word
                # .replace(" color", "")
                # .replace(" pattern", "")
                # .replace(" expression", "")
                # .replace(" type", "")
                # .replace(" length", "")
            )

            # Extend the maturity to include other words
            if att_list == 'young/baby':
                att_list += '/kid/kids/child/toddler/boy/girl'
            elif att_list == 'adult/old/aged':
                att_list += '/teen/elder'

            att_templates = []

            # Iterate synonyms of the current attribute
            for syn in att_list.split('/'):

                # For each synonym, iterate over the prompt types to use
                for prompt in use_prompts:

                    # For each prompt type, iterate over the templates available for it
                    for template in object_attribute_templates[is_has][prompt]:

                        # Type of attribute
                        # Attributes can be either 'has' (the attribute refers to an object possessed by
                        # the target) or 'is' (the attribute refers to the target itself)
                        if is_has == 'has':
                            att_templates.append(
                                template.format(
                                    attr=syn, dobj=dobj_name, noun=object_word
                                ).strip()
                            )
                        elif is_has == 'is':
                            att_templates.append(
                                template.format(attr=syn, noun=object_word).strip()
                            )

            # Store all the prompts generated for current attribute
            all_att_templates.append(att_templates)

        att_templates_syn = all_att_templates
        # Create a list that maps index to attribute id (since the same attribute appears multiple times
        # due to templating). This allows to split the list/tensor later
        len_synonyms = [len(att_synonyms) for att_synonyms in all_att_templates]
        att_ids = [
            [att_dict['id']] * len(att_synonyms)
            for att_dict, att_synonyms in zip(
                attributes, att_templates_syn
            )
        ]
        att_ids = list(itertools.chain.from_iterable(att_ids))
        all_att_templates = list(itertools.chain.from_iterable(all_att_templates))
        text_features = torch.cat([
            clip_encode_text(all_att_templates[batch_start:batch_start+batch_size], model['model'], model['tokenizer'], device, model_from=model['model_from'], normalize=normalize, padding=padding)
            for batch_start in range(0, len(all_att_templates), batch_size)
        ], dim=0)

        att_emb = {
            'att_cls': [att['name'] for att in attributes],
            'len_synonyms': len_synonyms,
            'att_ids': att_ids,
            'all_att_templates': all_att_templates,
            'text_features': text_features.cpu().numpy(),
        }

        if save_result:
            print('Saving attribute text embeddings at', att_emb_path)
            with open(att_emb_path, 'wb') as t:
                pickle.dump(att_emb, t)

    if return_prompts:
        return text_features, len_synonyms, all_att_templates
    else:
        return text_features, len_synonyms
