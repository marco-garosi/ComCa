import torch

def load_cache(
        model_name,
        prefix, suffix,
        shots,
        categories,
        path_to_cache,
        cache_by_category=False,
        soft_labels=None,
        device='cpu'
    ):
    # Load TIP-Adapter Cache
    # "Hierarchical" cache, i.e. cache by category (each category has its own cache)
    if cache_by_category:
        cache_keys = {}
        cache_values = {}
        for category_idx, category in categories.iterrows():
            prefix_cat = prefix + '_' + category['name'].replace(' ', '_')
            cache_keys[category['id']] = torch.load(path_to_cache.format(prefix=prefix_cat, t='keys', model=model_name, shots=shots, suffix=suffix), map_location=torch.device(device)).float().to(device)
            
            suffix_values = suffix
            if soft_labels is not None and len(soft_labels) > 0:
                suffix_values += + f'__{soft_labels}'
            cache_values[category['id']] = torch.load(path_to_cache.format(prefix=prefix_cat, t='values', model=model_name, shots=shots, suffix=suffix_values), map_location=torch.device(device)).float().to(device)

    # Plain cache
    else:
        cache_keys = torch.load(path_to_cache.format(prefix=prefix, t='keys', model=model_name, shots=shots, suffix=suffix), map_location=torch.device(device)).float().to(device)
        if soft_labels is not None and len(soft_labels) > 0:
            suffix += f'__{soft_labels}'
        cache_values = torch.load(path_to_cache.format(prefix=prefix, t='values', model=model_name, shots=shots, suffix=suffix), map_location=torch.device(device)).float().to(device)

    print("Cache path:", path_to_cache.format(prefix=prefix, t='values', model=model_name, shots=shots, suffix=suffix))

    return cache_keys, cache_values
