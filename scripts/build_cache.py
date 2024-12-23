import sys
sys.path.append('./')

import warnings
warnings.filterwarnings('ignore')

import os
import copy
import json
import itertools
import numpy as np
import argparse
from collections import Counter
from PIL import Image
from source.utils.annotations_loading import load_annotations
from source.utils import (
    load_model,
    get_device,
    load_templates_cache_generation,
)
from source.model.cache_utils import *
from source.utils.encode_clip import *
from source.cache_generation import parse_attribute
from source.utils.distributions import *
from source.tip_adapter.build_cache import *
from source.data import get_ovad_dataloader, get_dataloader, MemoryMappedTensorDataset
from source.utils import RetrievalDatabase, RetrievalVocabulary
import torchvision.transforms as transforms
import config


# Configuration
RETRIEVAL_DATASET = 'cc12m'
cfg = {
    'model_name': 'ViT-B-32',
    'pretrained': 'laion2b_e16',
    'model_from': 'OpenCLIP',
    'load_cache': False,
    'prefix': '',
    'cache_dir': os.path.join(config.OUT_PATH, 'comca-cache'),
    'augment_epoch': 1,
    'shots': 16,
    'replace': False,
    'batch_size': 256,
    'shuffle': False,
    'scalings': ['given_mean_and_std_by_object',],
    'T_sharpen': 2.0,  # used only for sharpening (ablation)

    'base_cache_dir': os.path.join(config.OUT_PATH, 'comca-cache'),
    'model_cache_dir': config.MODEL_CACHE_DIR,
}
cfg['model_name_slug'] = cfg['model_from'].lower() + '__' + cfg['model_name'].replace('-', '_').replace('/', '_') + '__' + cfg['pretrained']
original_cfg = copy.deepcopy(cfg)

# Model used for retrieval
image_encoding_model_cfg = copy.deepcopy(cfg)
image_encoding_model_cfg['model_name'] = 'ViT-B-32'
image_encoding_model_cfg['pretrained'] = 'laion2b_e16'
image_encoding_model_cfg['model_from'] = 'OpenCLIP'
image_encoding_model_cfg['model_name_slug'] = image_encoding_model_cfg['model_from'].lower() + '__' + image_encoding_model_cfg['model_name'].replace('-', '_').replace('/', '_') + '__' + image_encoding_model_cfg['pretrained']
image_encoding_model_cfg['image_size'] = 224  # 384 for BLIP

# Soft labels name
path = '{prefix}_{t}__{model}__{shots}_shots{suffix}.pt'

channel_stats = dict(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

transform = transforms.Compose(
    [
        transforms.Resize(size=(image_encoding_model_cfg['image_size'], image_encoding_model_cfg['image_size']), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats),
    ]
)


def get_arguments():
    parser = argparse.ArgumentParser(description='open_clip evaluation')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('--distribution', type=str)
    parser.add_argument('--reweighting', type=str, default=None)
    parser.add_argument('--retrieval_modality', type=str, default='image')
    parser.add_argument('--prompt_bs', type=int, default=512)
    parser.add_argument('--embedding_model_bs', type=int, default=64)
    parser.add_argument('--template_file', type=str, default='OvarNet_with_category.json')
    parser.add_argument('--template', type=str, default='none')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='top-0.8_weighted')
    parser.add_argument('--alpha', type=float, default=0.6)

    return parser.parse_args()


def get_annotations(dataset):
    all_annotations = load_annotations('.', dataset=dataset)

    if dataset == 'OVAD':
        # attr_list = attribute_name:attribute_1/attribute_2/...
        raw_attributes = [attr_list.split(':')[1].split('/') for attr_list in all_annotations['idx2attr'].values()]
        # Flatten the list
        raw_attributes = list(itertools.chain.from_iterable(raw_attributes))

        idx2is_has = all_annotations['idx2is_has']
    else:
       raw_attributes = list(all_annotations['idx2attr'].values())
       idx2is_has = {k: 'is' for k in all_annotations['idx2attr'].keys()}

    return all_annotations, {
        'raw_attributes': raw_attributes,
        'idx2is_has': idx2is_has,
    }


def t2x(x):
    return 't2t' if x == 'text' else 't2i'


def get_initial_withdrawal(args) -> int:
    if args.dataset == 'OVAD':
        return 117
    elif args.dataset == 'VAW':
        return 256
    
    return 64


def main(args):
    global cfg
    global original_cfg
    global image_encoding_model_cfg
    global channel_stats
    global transform
    global path

    cfg['cache_dir'] = os.path.join(cfg['cache_dir'], args.dataset)
    original_cfg['cache_dir'] = os.path.join(original_cfg['cache_dir'], args.dataset)
    image_encoding_model_cfg['cache_dir'] = os.path.join(image_encoding_model_cfg['cache_dir'], args.dataset)

    # Get the device
    device = get_device()

    # Load the model
    print('Loading model')
    model, processor, tokenizer = load_model(cfg['model_from'], cfg['model_name'], pretrained=cfg['pretrained'], device=device, cache_dir=cfg['model_cache_dir'])
    if image_encoding_model_cfg['model_name_slug'] == cfg['model_name_slug']:
        image_encoding_model = model
        image_encoding_processor = processor
        image_encoding_tokenizer = tokenizer
        same_model = True
    else:
        image_encoding_model, image_encoding_processor, image_encoding_tokenizer = load_model(image_encoding_model_cfg['model_from'], image_encoding_model_cfg['model_name'], pretrained=image_encoding_model_cfg['pretrained'], device=device, cache_dir=cfg['model_cache_dir'])
        same_model = False
    model.eval()
    image_encoding_model.eval()
    print('Model loaded')

    all_annotations, data = get_annotations(dataset=args.dataset)
    idx2is_has = data['idx2is_has']
    raw_attributes = data['raw_attributes']

    print('Loading scores')
    distribution, distribution_as_tensor = get_distribution(os.path.join(args.dataset, args.distribution), all_annotations, load_as='json', device=device)

    if args.reweighting:
        dist_reweighting, distribution_as_tensor_reweighting = get_distribution(os.path.join(args.dataset, args.reweighting), all_annotations, load_as='json', device=device)
        
        distribution_as_tensor_reweighted = {}
        for attribute_idx in distribution_as_tensor.keys():
            distribution_as_tensor_reweighted[attribute_idx] =\
                distribution_as_tensor[attribute_idx] *\
                distribution_as_tensor_reweighting[attribute_idx]
    print('Scores loaded')


    # Load the retrieval database
    subfolder = cfg['model_name'] + '__' + cfg['pretrained']
    database = RetrievalDatabase(os.path.join(config.OUT_PATH, RETRIEVAL_DATASET, subfolder, 'index'))

    # Load the embeddings
    print('Loading embeddings')
    dataset_image_embedding = MemoryMappedTensorDataset(os.path.join(config.OUT_PATH, 'vocabulary_free', 'all_images', RETRIEVAL_DATASET, f'{cfg["model_name_slug"]}.npy'))
    print('Embeddings loaded')

    # Retrieval modality
    subfolder = 'retrieved_from_' + RETRIEVAL_DATASET.replace('/', '_') + '_faiss' + '_' + t2x(args.retrieval_modality) + f'_{args.template}'
    cfg['cache_dir'] = os.path.join(cfg['base_cache_dir'], args.dataset)

    # Load templates
    templates = load_templates_cache_generation(args.template_file, config.BASE_PATH)

    # Retrieve and generate cache

    use_reweighted_tensor = args.reweighting is not None
    random_seed = args.seed
    mode = args.mode
    cfg = original_cfg

    all_keys, all_values, all_sampled_images = [], [], []
    prompts = []

    if categories := all_annotations.get('categories_by_split'):
        categories = categories.get('test')
    else:
        categories = all_annotations['categories']

    reset_seeds(random_seed)
    for attribute in all_annotations['idx2attr'].keys():
        if use_reweighted_tensor:
            distribution_to_use = distribution_as_tensor_reweighted[attribute]
        else:
            distribution_to_use = distribution_as_tensor[attribute]

        bindings = sample_category(categories, distribution_to_use, cfg['shots'], mode)
        bindings = Counter(bindings)

        for category, samples in bindings.items():
            category = category.replace(' ', '_')
            
            # Parse attribute
            group, synonyms = parse_attribute(attribute, all_annotations['idx2attr'])
            # Generate prompts
            is_has = idx2is_has[int(attribute)]
            for _ in range(samples):
                prompt = np.random.choice(templates[is_has][args.template])
                prompts.append(prompt.format(attr=', '.join(synonyms), noun=category, dobj=group))

        all_values.append(torch.tensor([int(attribute)] * cfg['shots']))

    repeat_prompt = [1 for _ in range(len(prompts))]

    print('Embedding prompts for retrieval')
    attributes_queries = torch.cat([
        clip_encode_text(prompts[batch_start:batch_start+args.prompt_bs], model, tokenizer, device=device, model_from=cfg['model_from'], normalize=True).cpu()
        for batch_start in range(0, len(prompts), args.prompt_bs)
    ], dim=0)
    attributes_queries = attributes_queries.cpu().numpy()
    print('Prompts for retrieval embedded')

    print('Retrieving samples')
    sampled_indices, image_paths = retrieve_shots_for_attributes(attributes_queries, database, repeat_prompt, modality=args.retrieval_modality, initial_withdrawl=get_initial_withdrawal(args), max_bs=2048, return_image_paths=not same_model)
    sampled_indices = sampled_indices.view(-1)
    print('Samples retrieved')

    # Free some memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if same_model:
        keys = dataset_image_embedding[sampled_indices]
    else:
        images = [
            Image.open(f'{config.PATH_TO_DATASET_SOURCE_IMAGES}/{path}.jpg')
            for path in image_paths
        ]
        
        if image_encoding_model_cfg['model_from'] in ['BLIP', 'X_VLM']:
            images = torch.stack([transform(image) for image in images])

        keys = torch.cat([
            clip_encode_images(
                image_encoding_model, image_encoding_processor,
                images[batch_start:batch_start+args.embedding_model_bs],
                model_from=image_encoding_model_cfg['model_from'],
                device=device,
                normalize=True,
            ).cpu()
            for batch_start in range(0, len(images), args.embedding_model_bs)
        ], dim=0)

    values = torch.cat(all_values)
    sampled_images = sampled_indices
    train_loader_cache = get_train_loader(False, keys, values, cfg, sampled_images=sampled_images)


    ###############
    # Build cache #
    ###############

    cfg = image_encoding_model_cfg
    cfg['suffix'] = f'seed_{random_seed}'
    cfg['encode_images'] = False
    cfg['augment_epoch'] = 1

    if cfg['encode_images']:
        cfg['suffix'] += '_with_transforms__augment_' + str(cfg["augment_epoch"])
        cfg['augment_epoch'] = 10


    cfg['prefix'] = subfolder + '_sampled'
    cfg['prefix'] += f'_from_{args.distribution.replace("/", "_")}'
    if use_reweighted_tensor:
        cfg['prefix'] += f'_reweighted_with_{args.reweighting.replace("/", "_")}'
    if len(mode) > 0:
        cfg['prefix'] += f'_{mode}'

    # Build and store the cache
    cache_keys, cache_values = build_cache_model_with_features(cfg, train_loader_cache, clip_model=model, data_is_normalized=True, device=device)

    # Generate soft labels
    attributes_embeddings = embed_attributes_for_soft_cache(
        args.dataset,
        model, tokenizer, cfg['model_from'],
        base_dir='.',
        device=device,
    )
    if args.dataset == 'VAW': cfg['scalings'].append('softmax')
    cache_keys = cache_keys.T
    cache_keys = cache_keys.to(device).float()
    cache_values = cache_values.to(device).float()
    cache_values = get_soft_labels(
        attributes_embeddings,
        cache_keys,
        cfg['scalings'],
        cache_values=cache_values,
        model_name=cfg['model_name'],
        model=model,
        device=device,
        alpha=args.alpha,
    )

    # Store soft labels
    suffix = get_suffix_cache(cfg, args.alpha)
    path = os.path.join(cfg['cache_dir'], path.format(prefix=cfg['prefix'], t='values', model=cfg['model_name_slug'], shots=str(cfg['shots']), suffix=suffix))
    torch.save(cache_values.cpu(), path)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
