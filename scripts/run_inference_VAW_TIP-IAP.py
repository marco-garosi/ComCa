import sys
sys.path.append('./')

import warnings
warnings.filterwarnings('ignore')

import os
import glob

import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

from ovad.ovad_evaluator import print_metric_table
from source.eval.VAW import Evaluator, eval

from source.utils.encode_clip import *
from source.utils.reproducibility import *
from source.utils import (
    get_device,
    load_model,
    load_templates_cache_generation,
)
from source.tip_adapter import load_cache
from source.model import (
    Predictor, PredictorWithCache, Cache,
    embed_prompts,
    HHVFMapper
)
from source.data import get_dataloader


# Constants to define behaviour
CACHE_TEMPLATE = '{prefix}_{t}__{model}__{shots}_shots{suffix}.pt'
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1
UNKNOWN_LABEL = 2


def get_arguments():
    parser = argparse.ArgumentParser(description='open_clip evaluation')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='VAW',
        help='dataset name',
    )
    parser.add_argument(
        '--ann_path',
        type=str,
        default='res/VAW',
        help='dataset name',
    )
    parser.add_argument(
        '--ann_file',
        type=str,
        default='ovad/ovad2000.json',
        help='annotation file with images and objects for attribute annotation',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out/VAW/multimodal_ovac/open_clip_ours/',
        help='dir where models are',
    )
    parser.add_argument(
        '--dir_data',
        type=str,
        default='datasets/VAW',
        help='image data dir',
    )
    parser.add_argument(
        '--model_from',
        type=str,
        default='OpenCLIP',  # 'OpenCLIP' # 'HF'
        help='where to load the model from',
    )
    parser.add_argument(
        '--model_arch',
        type=str,
        default='ViT-B/16',  # 'RN50' #'RN101' #'RN50x4' #'ViT-B/32' #'ViT-B/16'
        help='architecture name',
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default='laion400m_e31',  # 'cc12m' #'yfcc15m' #'openai' #'laion400m_e31' #'laion400m_e32' #'laion2b_e16'
        help='dataset trained name',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='all',
        help='prompt',
    )
    parser.add_argument(
        '--average_syn',
        action='store_true',
    )
    parser.add_argument(
        '--object_word',
        action='store_true',
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.17,
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1.,
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='sd'
    )
    parser.add_argument(
        '--shots',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--scale_logits',
        type=str,
        default='max',
    )
    parser.add_argument(
        '--augmented_cache',
        type=str,
        default=None
    )
    parser.add_argument(
        '--softmax',
        type=str,
        default='none'
    )
    parser.add_argument(
        '--soft_labels',
        type=str,
        default=''
    )
    parser.add_argument(
        '--scale_base_logits',
        type=float,
        default=1.
    )
    parser.add_argument(
        '--cache_by_category',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--cache_by_sample',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--path_for_image_features',
        type=str
    )
    parser.add_argument(
        '--path_to_cache',
        type=str,
        default=''
    )
    parser.add_argument(
        '--evaluate_metrics',
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        '--store_predictions',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--use_cache',
        action=argparse.BooleanOptionalAction,
        default=True
    )
    parser.add_argument(
        '--default_is_has',
        type=str,
        default='is'
    )
    parser.add_argument(
        '--template_file',
        type=str,
        default='OvarNet.json'
    )
    parser.add_argument(
        '--model_cache_dir',
        type=str,
        default=None
    )

    return parser.parse_args()


def main(args):
    device = get_device(allow_accelerator=True)

    # Templates
    object_word = 'object' if args.object_word else ''
    use_prompts = ['a', 'the', 'none']
    if args.prompt in use_prompts or args.prompt == 'photo':
        use_prompts = [args.prompt]

    # Get the folder for output and prepare for loading cache
    output_dir, att_emb_path, experiment_id = get_new_folder(args, object_word, max_id=20_000)
    print(f'Experiment ID: {experiment_id}')  # output now, so in case of crash we know where to look

    suffix = get_suffix(args)
    model_name = get_model_slug(args)

    # Store configuration in JSON file for reproducibility
    path_to_cache = os.path.join(args.path_to_cache, CACHE_TEMPLATE)
    save_args(args, script_name=os.path.basename(__file__), base_dir=output_dir)

    # Load annotations
    path = os.path.join(args.ann_path)
    with open(os.path.join(path, 'attr2idx.json'), 'r') as f:
        all_attributes = json.load(f)
    all_attributes_as_list = list(all_attributes.keys())
    annotations = [
        {
            'id': id,
            'name': attribute,
            'is_has_att': args.default_is_has,
        }
        for attribute, id in all_attributes.items()
    ]

    # Build Pandas DataFrame with categories
    with open(os.path.join(path, 'categories_by_split.json'), 'r') as f:
        categories = json.load(f)
    categories = pd.DataFrame.from_dict(categories['test']).rename(columns={0: 'name'})
    
    # Mapper
    mapper = HHVFMapper(all_attributes_as_list, mode='exact')

    # Load TIP-Adapter Cache
    cache_keys, cache_values = load_cache(model_name, args.prefix, suffix, args.shots, None, path_to_cache, cache_by_category=args.cache_by_category, soft_labels=args.soft_labels, device=device)
    cache = Cache(args.alpha, args.beta, cache_keys, cache_values, scale_logits='none', softmax='overall')

    # Load object-attribute distributions
    # Tensor has shape [#categories, #attributes]
    oa_distribution = torch.load(path_to_cache.format(prefix=args.prefix, t='attribute_dist', model=model_name, shots=args.shots, suffix=suffix), map_location=torch.device(device)).float().to(device)
    oa_distribution /= oa_distribution.sum(dim=-1, keepdim=True)  # normalize for each category, so that attributes within category sum to 1

    # Load backbone model
    model, processor, tokenizer = load_model(args.model_from, args.model_arch, pretrained=args.pretrained, device=device, cache_dir=args.model_cache_dir)

    # Data
    data_loader = get_dataloader(
        'VAW', images_path=args.dir_data, split='test',
        batch_size=args.batch_size,
        num_workers=os.cpu_count()
    )
    dataset = data_loader.dataset

    # Generate and embed (or load embeddings of) prompts
    templates = load_templates_cache_generation(args.template_file)['class_only']
    if args.prompt == 'all':
        templates = list(itertools.chain.from_iterable([list(x) for x in templates.values()]))
    else:
        templates = templates[args.prompt]
    categories_queries = []
    for category in categories['name'].tolist():
        # Generate prompts
        prompts = [
            template.format(noun=category)
            for template in templates
        ]

        category_queries = clip_encode_text(prompts, model, tokenizer, device=device, model_from=args.model_from, normalize=True)
        category_queries = category_queries.mean(dim=0)
        category_queries /= category_queries.norm(dim=-1)
        
        categories_queries.append(category_queries)
    text_features = torch.stack(categories_queries)

    # text_features = torch.load(path_to_cache.format(prefix=args.prefix, t='categories_queries', model=model_name, shots=args.shots, suffix=suffix), map_location=torch.device(device)).float().to(device)
    len_synonyms = [1] * len(text_features)

    pred_vectors = []
    label_vectors = []
    indx_max_syn = []

    # Load pre-computed image features for current model
    image_features_path = os.path.join(args.path_for_image_features, 'all_image_features', 'VAW', f'{model_name}.pt')
    try:
        all_image_features = torch.load(image_features_path, map_location=torch.device(device))
        all_image_features_found = True
    except:
        print('Image features will be generated and then saved')
        all_image_features = []
        all_image_features_found = False

    # Construct model for inference
    predictor = PredictorWithCache(
        model,
        args.model_from,
        processor,
        device,
        text_features,
        len_synonyms,
        cache=cache,
        average_syn=args.average_syn
    )
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader, desc='Batch')):
            batch_start = i * args.batch_size
            batch_end = min(len(dataset), i * args.batch_size + args.batch_size)
            
            # Classification labels
            obj_label = [x['object_name'] for x in labels]

            # Attribute labels
            att_labels = []
            for sample in labels:
                mapping_positive = mapper(sample['positive_attributes'])
                mapping_negative = mapper(sample['negative_attributes'])

                ground_truth = UNKNOWN_LABEL * np.ones(len(all_attributes), dtype=np.uint8)
                ground_truth[mapping_positive] = POSITIVE_LABEL
                ground_truth[mapping_negative] = NEGATIVE_LABEL
                att_labels.append(torch.from_numpy(ground_truth))
            att_labels = torch.stack(att_labels, axis=0)
            label_vectors.append(att_labels.cpu().numpy())

            # Predict
            image_features = all_image_features[batch_start:batch_end] if all_image_features_found else None
            output = predictor(images, scale_base_logits=args.scale_base_logits, obj_label=obj_label, image_features=image_features)
            image_features = output['image_features']
            x_attrs = output['scores']
            idx_attrs = output['idx_attrs']

            # IAP
            result = torch.zeros((len(x_attrs), 620)).to(device)
            for sample_x in range(len(x_attrs)):  # sample index
                result[sample_x] = oa_distribution.T @ x_attrs[sample_x]

                # The line above using the dot product is equivalent to the following two lines.
                # However, it runs much faster: the whole dataset is processed in 13 seconds with it,
                # while it takes 7 minutes 47 seconds with the following two lines. The following lines,
                # however, represent more closely the formula in the paper referenced in our work -- so
                # leaving them here for reference and to ease understanding!
                # for category in range(len(oa_distribution)):  # would be sum_{k=1}^{K} in paper
                #     result[sample_x] += oa_distribution[category] * x_attrs[sample_x][category]  # (620) * scalar
            x_attrs = result

            # Store image features if they were not found on disk, so that they
            # can be saved for future usage
            if not all_image_features_found:
                all_image_features.append(image_features)

            # Store to computer metrics
            pred_vectors.append(x_attrs.cpu().numpy())
            indx_max_syn.append(idx_attrs.cpu().numpy())
    
    # Store image features for future use without having to re-compute them
    if not all_image_features_found:
        all_image_features = torch.cat(all_image_features, dim=0)
        os.makedirs(os.path.dirname(image_features_path), exist_ok=True)
        torch.save(all_image_features, image_features_path)

    pred_vectors = np.concatenate(pred_vectors, axis=0)
    label_vectors = np.concatenate(label_vectors, axis=0)
    indx_max_syn = np.concatenate(indx_max_syn, axis=0)

    pred_vectors = np.nan_to_num(pred_vectors, nan=0)

    ###############
    #   Output    #
    ###############

    if args.evaluate_metrics:
        eval(pred_vectors, label_vectors, ann_path=args.ann_path, output_dir=output_dir, print_detailed_results=False)
    
    if args.store_predictions:
        np.save(os.path.join(output_dir, 'pred_vectors.npy'), pred_vectors)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
