import torch
import json
import pandas as pd
import numpy as np
import os


def load_annotations(base_dir='', dataset='OVAD', **kwargs):
    if dataset == 'OVAD':
        return load_annotations_ovad(base_dir=base_dir)
    if dataset == 'VAW':
        return load_annotations_vaw(base_dir=base_dir)

    raise Exception('`dataset` is not valid')


def load_annotations_ovad(base_dir=''):
    with open(os.path.join(base_dir, 'res', 'ovad', 'ovad2000.json'), 'r') as f:
        ovad_annotations = json.load(f)

    with open(os.path.join(base_dir, 'res', 'ovad', 'idx2attr.json'), 'r') as f:
        idx2attr = json.load(f)

    categories = pd.DataFrame.from_dict(ovad_annotations['categories'])
    category_id2idx = {id: idx for idx, id in enumerate(categories.id)}
    idx2is_has = {attr['id']: attr['is_has_att'] for attr in ovad_annotations['attributes']}

    classes_vector = torch.load(os.path.join(base_dir, 'res', 'ovad', 'classes_vector.pt'))
    label_vectors = torch.load(os.path.join(base_dir, 'res', 'ovad', 'label_vectors.pt'))
    raw_label_vectors = torch.load(os.path.join(base_dir, 'res', 'ovad', 'raw_label_vectors.pt'))

    groups = extract_groups(ovad_annotations)
    label_vectors_split = label_vectors.split(groups.tolist(), dim=-1)  # tuple with (#groups, #labels)
    where_to_split = groups.cumsum()
    map_to_group = lambda x: np.argmax(x < where_to_split)
    idx2group = torch.tensor([map_to_group(int(idx)) for idx in idx2attr.keys()])

    return {
        'ovad_annotations': ovad_annotations,
        'idx2attr': idx2attr,
        'idx2group': idx2group,
        'categories': categories,
        'category_id2idx': category_id2idx,
        'idx2is_has': idx2is_has,
        'classes_vector': classes_vector,
        'label_vectors': label_vectors,
        'raw_label_vectors': raw_label_vectors,
        'groups': groups,
        'label_vectors_split': label_vectors_split,
        'map_to_group': map_to_group,
    }


def load_annotations_vaw(base_dir=''):
    with open(os.path.join(base_dir, 'res', 'VAW', 'attr2idx.json'), 'r') as f:
        attr2idx = json.load(f)
    idx2attr = {v: k for k, v in attr2idx.items()}

    with open(os.path.join(base_dir, 'res', 'VAW', 'categories_by_split.json'), 'r') as f:
        categories_by_split = {
            split: pd.DataFrame(categories).rename(columns={0: 'name'})
            for split, categories in json.load(f).items()
        }
    with open(os.path.join(base_dir, 'res', 'VAW', 'all_categories.json'), 'r') as f:
        categories = pd.DataFrame(json.load(f)).rename(columns={0: 'name'})

    return {
        'attr2idx': attr2idx,
        'idx2attr': idx2attr,
        'categories_by_split': categories_by_split,
        'categories': categories,
    }


def extract_groups(annotations):
    attributes_for_category = {}
    attributes_for_category_as_text = {}

    for att_dict in annotations['attributes']:
        att_w_type = att_dict['name']
        att_type, att_list = att_w_type.split(':')
        
        attributes_for_category[att_type] = 1 + attributes_for_category.get(att_type, 0)
        attributes_for_category_as_text[att_type] = attributes_for_category_as_text.get(att_type, []) + att_list.split('/')[0:1]

    return np.array([v for v in attributes_for_category.values()])
