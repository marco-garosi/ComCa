from typing import Optional

import os
import yaml
import json
import random
import torch
import numpy as np
from source.utils import intersect_tensors, difference_tensors
from source.utils import RetrievalDatabase
from tqdm import tqdm

import config


def get_distribution(subfolder, all_annotations, load_as: str = 'yaml', save_as_json: bool = True, device='cpu', base: str = '.'):
    distribution = {}
    distribution_as_tensor = {}

    if categories := all_annotations.get('categories_by_split'):
        categories = categories.get('test')
    else:
        categories = all_annotations['categories']
    category_slugs = categories['name'].tolist()

    for attribute_idx in all_annotations['idx2attr'].keys():
        agnostic_path = os.path.join(base, 'res', 'category_scores', subfolder, f'{attribute_idx}')

        if load_as == 'yaml':
            path = agnostic_path + '.yml'
            with open(path, 'r') as f:
                data = yaml.safe_load(f)

            if save_as_json:
                path = agnostic_path + '.json'
                with open(path, 'w') as f:
                    json.dump(data, f)
        
        elif load_as == 'json':
            path = agnostic_path + '.json'
            with open(path, 'r') as f:
                data = json.load(f)

        distribution[attribute_idx] = {
            k: data[k]
            for k in category_slugs
        }
            
        distribution_as_tensor[attribute_idx] = torch.tensor([v for v in distribution[attribute_idx].values()]).to(device).float()

    return distribution, distribution_as_tensor


def sample_category(categories, distribution, samples=1, mode='', device='cpu'):
    category = None
    distribution = torch.clone(distribution).to(device)

    if distribution.sum() <= 0.99:
        return categories.iloc[np.random.choice(np.arange(len(categories)), samples)]['name'].tolist()
    
    if 'top-' in mode:
        mode, topk = mode.split('top-')
        topk, topk_mode = topk.split('_')
        topk = float(topk)

        if topk.is_integer():
            values, indices = distribution.topk(int(topk))
            indices = indices[values > 0]  # Filter only values that are > 0, as this could cause unintended behavior when sampling

        # Top-k with thresholding, i.e. k is determined dynamically
        else:
            distribution /= distribution.sum()  # normalize first
            values, indices = distribution.sort(descending=True)
            sum, chosen = 0, []
            for index, value in zip(indices, values):
                sum += value
                chosen.append(index)
                # topk is our threshold
                if sum >= topk:
                    break
            indices = torch.tensor(chosen)

        # Common to both cases:
        # - set all values to -inf
        # - keep probabilities of the topk attributes
        indices = indices.to(device)
        mask = torch.ones(len(distribution)).to(device) * float('-inf')
        mask[indices] = distribution[indices]
        distribution = mask

        if topk_mode == 'weighted':
            # Prepare the distribution for subsequent sampling
            distribution = distribution.softmax(dim=0)
        elif topk_mode == 'random':
            # Randomly sample a category
            indices = (~distribution.isinf()).nonzero().view(-1)
            category = categories.iloc[np.random.choice(indices.cpu(), samples)]['name'].tolist()

    elif 'softmax' in mode:
        if 'norm' in mode:
            distribution /= distribution.sum()  # normalize first
            
        distribution = distribution.softmax(dim=0)

    elif mode == 'random_uniform':
        category = categories.iloc[np.random.choice(np.arange(0, len(categories)), samples)]['name'].tolist()

    else:
        distribution /= distribution.sum()
    
    if category is None:
        category = categories.iloc[np.random.choice(np.arange(0, len(categories)), samples, p=distribution.cpu().numpy())]['name'].tolist()

    return category


def retrieve_shots_for_attributes(
    attribute_query: torch.Tensor,
    database: RetrievalDatabase,
    shots: list[int],
    modality: str = 'text',
    exclude: Optional[torch.Tensor] = torch.tensor([]),
    initial_withdrawl: int = 117,
    max_bs: int = 1024,
    return_image_paths: bool = False,
    deduplication: bool = True,
):
    # Sample more than `shots` to ensure the possibility to exlude tensors
    # Typically, sampling 117 times more should be enough, as this would allow the code to sample a different
    # image every time, even if all the attributes had the exact same matches (virtually impossible)
    safe_withdrawl = initial_withdrawl

    if len(attribute_query) > max_bs:
        matches = []
        for batch_idx, batch_start in enumerate(tqdm(range(0, len(attribute_query), max_bs), desc='Retrieval batch')):
            matches += database.query(attribute_query[batch_start:batch_start + max_bs], modality=modality, num_samples=safe_withdrawl, return_metadata=return_image_paths, deduplication=deduplication)
    
    else:
        matches = database.query(attribute_query, modality=modality, num_samples=safe_withdrawl, return_metadata=return_image_paths, deduplication=deduplication)

    if exclude is None:
        exclude = torch.tensor([], dtype=int)
    exclude = torch.clone(exclude)

    all_sampled_indices = []
    image_paths = [] if return_image_paths else None
    for idx, match in enumerate(matches):
        num_samples_multiplyer = 2
        base = safe_withdrawl
        while True:
            sampled_indices = torch.tensor([x['id'] for x in match])
            sampled_indices = difference_tensors(sampled_indices, exclude)[:shots[idx]]

            if len(sampled_indices) < shots[idx]:
                match = database.query(attribute_query[idx:idx+1], modality=modality, num_samples=base * num_samples_multiplyer, return_metadata=return_image_paths, deduplication=deduplication)[0]
                num_samples_multiplyer += 1
                continue

            if return_image_paths:
                for sampled_index in sampled_indices:
                    res = [x['image_path'] for x in match if x['id'] == sampled_index and x.get('image_path') is not None]
                    if len(res) > 0:
                        image_paths.append(res[0])

            all_sampled_indices.append(sampled_indices)
            exclude = torch.cat([exclude, sampled_indices])
            break

    return torch.cat(all_sampled_indices), image_paths


def reset_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_train_loader(use_raw_images, keys, values, cfg, sampled_images=None, cache_transform=None):
    if use_raw_images:
        # Build data loader
        train_loader_cache = torch.utils.data.DataLoader(
            DatasetWrapper([{
                'impath': file_list[idx.item()],
                'label': label_vectors[idx.item()]
            } for idx in sampled_images],
            input_size=224, transform=cache_transform, is_train=True),
            batch_size=cfg['batch_size'],
            # num_workers=8,
            shuffle=cfg['shuffle'],
            drop_last=False,
            # pin_memory=(torch.cuda.is_available())
        )

    else:
        train_loader_cache = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(keys, values),
            batch_size=cfg['batch_size'],
            shuffle=cfg['shuffle']
        )

    return train_loader_cache
