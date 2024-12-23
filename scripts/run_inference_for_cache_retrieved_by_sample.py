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

from ovad.misc import ovad_validate

from source.utils.reproducibility import *
from source.utils import (
    get_device,
    load_model,
    load_templates_cache_generation,
    RetrievalDatabase,
    clip_encode_images,
)
from source.data import get_ovad_dataloader, MemoryMappedTensorDataset
from source.tip_adapter import load_cache
from source.model import (
    Predictor, PredictorWithCache, Cache,
    embed_prompts
)
from source.model.cache_utils import *
import config


CACHE_TEMPLATE = '{prefix}_{t}__{model}__{shots}_shots{suffix}.pt'


def get_arguments():
    parser = argparse.ArgumentParser(description='open_clip evaluation')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ovad2000',
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
        default='out/OVAD_Benchmark2000/multimodal_ovac/open_clip_ours/',
        help='dir where models are',
    )
    parser.add_argument(
        '--dir_data',
        type=str,
        default='datasets/ovad_box_instances/2000_img',
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
        '--image_size',
        type=int,
        default=224
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
    object_word = 'object' if args.object_word else ''
    use_prompts = ['a', 'the', 'none']
    if args.prompt in use_prompts or args.prompt == 'photo':
        use_prompts = [args.prompt]

    object_attribute_templates = load_templates_cache_generation(args.template_file)

    # Get the folder for output and prepare for loading cache
    output_dir, att_emb_path, experiment_id = get_new_folder(args, object_word, max_id=20_000)
    print(f'Experiment ID: {experiment_id}')  # output now, so in case of crash we know where to look

    suffix = get_suffix(args)
    model_name = get_model_slug(args)

    # Store configuration in JSON file for reproducibility
    path_to_cache = os.path.join(args.path_to_cache, CACHE_TEMPLATE)
    store_config(args, path_to_cache, output_dir)

    # Load annotations
    annotations = json.load(open(args.ann_file, 'r'))
    device = get_device(allow_accelerator=True)

    # Build Pandas DataFrame with categories
    categories = pd.DataFrame.from_dict(annotations['categories'])

    # Load TIP-Adapter Cache
    if args.use_cache:
        cache_keys, cache_values = None, None
        cache = Cache(args.alpha, args.beta, cache_keys, cache_values, scale_logits=args.scale_logits, softmax=args.softmax)
    else:
        cache = None

    # Load backbone model
    model, processor, tokenizer = load_model(args.model_from, args.model_arch, pretrained=args.pretrained, device=device, cache_dir=args.model_cache_dir)

    # Data
    data_loader = get_ovad_dataloader(
        args.dir_data, args.image_size, args.batch_size, 4
    )
    dataset = data_loader.dataset

    # Generate and embed (or load embeddings of) prompts
    text_features, len_synonyms = embed_prompts(
        att_emb_path,
        annotations,
        use_prompts, object_attribute_templates,
        {'model': model, 'tokenizer': tokenizer, 'model_from': args.model_from},
        object_word=object_word,
        save_result=True,
        allow_loading=True,
        normalize=True,
        device=device,
    )

    pred_vectors = []
    label_vectors = []
    indx_max_syn = []

    # Load pre-computed image features for current model
    image_features_path = os.path.join(args.path_for_image_features, 'all_image_features', f'{model_name}.pt')
    try:
        all_image_features = torch.load(image_features_path, map_location=torch.device(device))
        all_image_features_found = True
    except:
        print('Image features will be generated and then saved')
        all_image_features = []
        all_image_features_found = False

    # Construct model for inference
    predictor: torch.nn.Module = PredictorWithCache if args.use_cache else Predictor
    predictor = predictor(
        model,
        args.model_from,
        processor,
        device,
        text_features,
        len_synonyms,
        cache=cache,
        average_syn=args.average_syn
    )
    
    
    slug = args.model_arch + '__' + args.pretrained
    dataset_image_embedding = MemoryMappedTensorDataset(
        os.path.join(config.DATASET_PATH, 'vocabulary_free', 'all_images', 'cc12m', f'{model_name}.npy'),
        (10269691, 512),
        dtype=np.float16
    )
    attributes_embeddings = embed_attributes_for_soft_cache(
        'OVAD',
        model, tokenizer, args.model_from,
        base_dir='.',
        device=device,
    )
    SCALINGS = ['given_mean_and_std_by_object']

    database = RetrievalDatabase(os.path.join(config.DATASET_PATH, 'cc12m', f'{slug}', 'index'))


    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader)):
            if args.cache_by_sample:
                image_features = clip_encode_images(model, processor, images, model_from=args.model_from, device=device, normalize=True)
                matches = database.query(image_features.cpu().numpy(), modality='image', num_samples=args.shots * 32, return_metadata=False, deduplication=False)
                sampled_indices = torch.tensor([x['id'] for x in matches[0][:args.shots]])  # batch size must be 1!
                cache_keys = dataset_image_embedding[sampled_indices].to(device).float()
                cache_values = get_soft_labels(
                    attributes_embeddings,
                    cache_keys,
                    SCALINGS,
                    model_name=args.model_arch,
                    model=model,
                    device=device,
                )
                cache.keys = cache_keys.T
                cache.values = cache_values

            batch_start = i * args.batch_size
            batch_end = min(len(dataset), i * args.batch_size + args.batch_size)
            att_label, obj_label = torch.stack(labels[0], axis=1), labels[1]
            label_vectors.append(att_label.cpu().numpy())

            # Predict
            image_features = all_image_features[batch_start:batch_end] if all_image_features_found else None
            output = predictor(images, scale_base_logits=args.scale_base_logits, obj_label=obj_label, image_features=image_features)
            image_features = output['image_features']
            x_attrs = output['scores']
            idx_attrs = output['idx_attrs']

            # Store image features if they were not found on disk, so that they
            # can be saved for future usage
            if not all_image_features_found:
                all_image_features.append(image_features.cpu())

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

    ###############
    #   Output    #
    ###############

    if args.evaluate_metrics:
        ovad_validate(
            annotations['attributes'],
            pred_vectors,
            label_vectors,
            output_dir,
            args.dataset_name,
            save_output=False,
            print_table=False
        )
    
    if args.store_predictions:
        np.save(os.path.join(output_dir, 'pred_vectors.npy'), pred_vectors)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
