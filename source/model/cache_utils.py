import torch


def sharpen(p, T, dim=1):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=dim, keepdim=True)
    return sharp_p


def scale(blended_labels, mode, **kwargs):
    if mode == 'minmax':
        blended_labels -= blended_labels.min()
        blended_labels /= blended_labels.max()
    elif mode == 'minmax_row':
        blended_labels -= blended_labels.min(dim=-1, keepdim=True).values
        blended_labels /= blended_labels.max(dim=-1, keepdim=True).values
    elif mode == 'minmax_col':
        blended_labels -= blended_labels.min(dim=0, keepdim=True).values
        blended_labels /= blended_labels.max(dim=0, keepdim=True).values
    elif mode == 'exp_sharpen':
        blended_labels = blended_labels.exp()
    elif mode == 'softmax':
        if kwargs.get('softmax_set_-inf', True):
            blended_labels[blended_labels == 0.0] = float('-inf')
        blended_labels *= kwargs.get('softmax_temp', 1.0)
        blended_labels = blended_labels.softmax(dim=-1)
    elif mode == 'softmax_col':
        if kwargs.get('softmax_set_-inf', True):
            blended_labels[blended_labels == 0.0] = float('-inf')
        blended_labels *= kwargs.get('softmax_temp', 1.0)
        blended_labels = blended_labels.softmax(dim=0)
    elif mode == 'sharpen':
        blended_labels = sharpen(blended_labels, T)
    elif mode == 'sharpen_col':
        blended_labels = sharpen(blended_labels, T, dim=0)
    elif mode == 'given_mean_and_std_by_object':
        mean = blended_labels.mean(dim=-1, keepdim=True)
        std = blended_labels.std(dim=-1, keepdim=True)
        blended_labels = kwargs['target_mean'] + (blended_labels - mean) * (kwargs['target_std'] / std)

    return blended_labels


def get_soft_labels(
    attributes_embeddings,
    cache_keys,
    scalings,
    cache_values = None,
    target_mean: float = 0.2064,
    target_std: float = 0.0438,
    softmax_temp: float = 7.0,
    device = 'cpu',
    model_name: str = 'siglip',
    model = None,
    alpha: float = 0.6,
):
    soft_labels = (attributes_embeddings.to(device) @ cache_keys.T).T

    if 'siglip' in model_name:
        soft_labels *= model.logit_scale.exp()
        soft_labels += model.logit_bias
        soft_labels = torch.sigmoid(soft_labels)

    if cache_values is not None:
        blended_labels = alpha * soft_labels + (1 - alpha) * cache_values
    else:
        blended_labels = soft_labels

    for scaling in scalings:
        blended_labels = scale(blended_labels, scaling, target_mean=target_mean, target_std=target_std, softmax_temp=softmax_temp)

    return blended_labels


def get_idx2attr(dataset: str):
    import os
    import config
    import json

    if dataset == 'OVAD':
        with open(os.path.join(config.RES_PATH, 'ovad', 'idx2attr.json')) as f:
            return json.load(f)

    elif dataset == 'VAW':
        with open(os.path.join(config.RES_PATH, 'VAW', f'attr2idx.json'), 'r') as f:
            attr2idx = json.load(f)
            return {str(v): k for k, v in attr2idx.items()}

    return None


def get_idx2is_has(dataset: str, path: str = '..', default_is_has: str = 'is', idx2attr = None):
    from source.utils.annotations_loading import load_annotations

    if dataset == 'OVAD':
        all_annotations = load_annotations(path)
        return all_annotations['idx2is_has']
    
    elif dataset == 'VAW':
        return {int(k): default_is_has for k in idx2attr.keys()}

    return None


def embed_attributes_for_soft_cache(
    dataset: str,
    model, tokenizer, model_from,
    use_template: str = 'a',
    object_word: str = 'object',
    template_file: str = 'ovad.json',
    base_dir: str = '..',
    device: str = 'cpu',
):
    from source.utils import load_templates_cache_generation
    from source.cache_generation import parse_attribute
    from source.utils import load_templates_cache_generation
    from source.utils import clip_encode_text

    templates = load_templates_cache_generation(template_file, base_dir=base_dir)
    idx2attr = get_idx2attr(dataset)
    idx2is_has = get_idx2is_has(dataset, path=base_dir, idx2attr=idx2attr)

    attributes_embeddings = []
    for attribute in idx2attr.keys():
        attribute_type, synonyms = parse_attribute(attribute, idx2attr)

        prompts = []
        for synonym in synonyms:
            for template in templates[idx2is_has[int(attribute)]][use_template]:
                prompts.append(template.format(attr=synonym, dobj=attribute_type, noun=object_word))

        with torch.no_grad():
            text_features = clip_encode_text(prompts, model, tokenizer, device=device, model_from=model_from, normalize=True)
        text_features = text_features.mean(dim=0)

        attributes_embeddings.append(text_features)

    attributes_embeddings = torch.stack(attributes_embeddings, dim=0)
    
    return attributes_embeddings


def get_suffix_cache(cfg, alpha):
    if cfg['suffix'] is not None and len(cfg['suffix']) > 0:
        suffix = '_' + cfg['suffix']
    else:
        suffix = ''

    soft_labels_name = f'alpha_{alpha}'
    if len(cfg['scalings']) > 0:
        for scaling in cfg['scalings']:
            if scaling == 'softmax':
                soft_labels_name += '_s'
                continue
            
            soft_labels_name += f'_{scaling}'
            if scaling in ['sharpen', 'sharpen_col']:
                soft_labels_name += f'_T_{cfg["T_sharpen"]}'

    suffix += f'__{soft_labels_name}'

    return suffix
