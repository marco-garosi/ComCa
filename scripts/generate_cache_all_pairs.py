import torch

import json

import os
import sys
import argparse
from tqdm import tqdm

sys.path.append('./')
from source.cache_generation.generate_prompt import generate_prompt
from source.utils.pipeline_loading import load_diffusion_pipeline
from source.utils.annotations_loading import load_annotations
from source.utils.templating import load_templates_cache_generation
from source.utils.reproducibility import save_args
from source.cache_generation.image_batch_generation import generate_images_batch


MODE = 'uniform'


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--templates_file', type=str, default='ovad.json')

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--resume_from_epoch', type=int, default=0)
    parser.add_argument('--resume_from_category', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bs', type=int, default=1)

    parser.add_argument('--template', type=str, default='photo')
    parser.add_argument('--negative_prompts', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--scheduler', type=str, default='default')
    parser.add_argument('--num_inference_steps', type=int, default=20)

    parser.add_argument('--out_folder', type=str, default='sd')

    parser.add_argument('--choose_categories', type=str, default=None)

    return parser


def parse_args(parser=None):
    if parser is None:
        parser = get_parser()

    return parser.parse_args()


def generate(
        pipeline, distributed_state,
        seeds,
        idx2attr, idx2is_has,
        categories, templates,
        idx2group, negative_prompts,
        mode='uniform', template='photo',
        label_vectors=None, classes_vector=None,
        category_id2idx=None, on_empty_distribution='uniform',
        num_inference_steps=50, batch_size=1
    ):
    metadata_store = {}
    images_store = {}

    for attribute in tqdm(idx2attr.keys(), desc='Attribute'):
        metadata = generate_prompt(int(attribute), idx2attr, idx2group, idx2is_has, categories, templates, get_negative_attributes=negative_prompts, mode=mode, template=template, label_vectors=label_vectors, classes_vector=classes_vector, category_id2idx=category_id2idx, on_empty_distribution=on_empty_distribution)
        # This can happen when using conditional distributions to sample class bindings and `on_empty_distribution` is None
        if metadata is None:
            continue

        metadata['seed'] = seeds[int(attribute)].item()
        metadata_store[attribute] = metadata

    images_store = generate_images_batch(pipeline, distributed_state, seeds, metadata_store, num_inference_steps=num_inference_steps, batch_size=batch_size)

    return metadata_store, images_store


def main(args):
    # Load annotations
    print('=== Loading Annotations ===')
    annotations = load_annotations()
    templates = load_templates_cache_generation(args.templates_file)
    categories = annotations['categories']

    # Load model
    print('=== Loading Model ===')
    pipeline, distributed_state = load_diffusion_pipeline(args.model, args.scheduler, compile=args.compile)

    # Generate seeds for image generation
    print('=== Generating Seeds ===')
    torch.manual_seed(args.seed)
    seeds = torch.randint(0, int(1e5), (args.epochs, len(categories), len(annotations['idx2attr'])))

    # Resuming
    resume_from_category = None
    if args.resume_from_category is not None:
        resume_from_category = args.resume_from_category

    # Generate images
    print(f'Resuming from epoch {args.resume_from_epoch}')
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        if epoch < args.resume_from_epoch:
            continue

        print(f'=== Epoch {epoch} ===')

        # Iterate over categories
        for category_idx in range(len(categories)):
            current_category = categories.iloc[category_idx:category_idx+1]
            category_name = current_category.iloc[0]['name']
            category_slug = category_name.replace(' ', '_')
            folder = os.path.join('out', 'metadata', args.out_folder, category_slug)

            if os.path.exists(folder) and os.path.isfile(os.path.join(folder, f'metadata_store_epoch_{epoch}.json')):
                print(f'=== Skipping Category {category_name} ===')
                continue

            # Should resume from a category?
            if resume_from_category is not None:
                # Found the correct category -- so disable the resume option, so that it'll work
                # as usual from the next category (iteration)
                if resume_from_category == category_slug:
                    resume_from_category = None
                # Not yet found the right category, just go the next category (iteration)
                else:
                    continue

            print(f'=== Category {category_name} ===')

            metadata, images = generate(pipeline, distributed_state, seeds[epoch, category_idx], annotations['idx2attr'], annotations['idx2is_has'], current_category, templates, annotations['idx2group'], args.negative_prompts, mode=MODE, template=args.template, num_inference_steps=args.num_inference_steps, batch_size=args.bs, classes_vector=annotations['classes_vector'], label_vectors=annotations['label_vectors'], category_id2idx=annotations['category_id2idx'])
            
            # Store metadata for reproducibility
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, f'metadata_store_epoch_{epoch}.json'), 'w') as f:
                json.dump(metadata, f)

            # Store output images
            for attribute_idx, image in images.items():
                folder = os.path.join('out', 'images', args.out_folder, category_slug, str(attribute_idx))
                os.makedirs(folder, exist_ok=True)
                image.save(os.path.join(folder, f'epoch_{epoch}.png'))


if __name__ == '__main__':
    args = parse_args()
    if args.model is None:
        raise Exception('Model should not be None')
    save_args(args, script_name=sys.argv[0])

    main(args)
