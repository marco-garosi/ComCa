from typing import Union

import torch
import numpy as np
import pandas as pd
import itertools


def parse_attribute(attribute_idx: Union[int, str], idx2attr: dict):
    """Parse an attribute into group and synonyms

    Args:
        attribute_idx (Union[int, str]): index of the attribute to parse
        idx2attr (dict): dictionary of all attributes

    Returns:
        (str, list[str]): (group, list of synonyms)
    """

    # Extract attribute
    if isinstance(attribute_idx, torch.Tensor):
        attribute_idx = attribute_idx.item()

    return parse_attribute_from_string(idx2attr.get(str(attribute_idx), ':'))


def parse_attribute_from_string(text: str):
    """Parse an attribute, provided as a string, into group and synonyms

    Args:
        text (str): the attribute, in the form of "group:synonym_1/synonym_2/..."

    Returns:
        (str, list[str]): (group, list of synonyms) if parsing is successful, (None, [`text`]) otherwise
    """

    try:
        group, synonyms = text.split(':')
        synonyms = synonyms.split('/')

        return group, synonyms
    
    except:
        return None, [text]


def generate_prompt_uniform(attribute_idx, idx2attr, idx2is_has, categories, templates, template='none'):
    """Generate a prompt by binding the attribute to a uniformly sampled class

    Args:
        attribute_idx (int): index of the attribute to generate the prompt for
        idx2attr (dict): dictionary mapping an index to the textual attribute
        idx2is_has (dict): dictionary mapping an attribute (by its index) to its type, either 'is' or 'has'
        categories (pd.DataFrame): dataframe with all categories and corresponding ids
        templates (dictionary): templates to use to generate prompts
        template (str, optional): which type of template to use. Defaults to 'none'.

    Returns:
        dict: the bound category, the group of the attribute, the attribute itself, the prompt
    """

    group, synonyms = parse_attribute(attribute_idx, idx2attr)

    # Sample a category, randomly
    category = categories.iloc[np.random.choice(np.arange(0, len(categories)))]['name']
    
    # Generate prompt
    is_has = idx2is_has[attribute_idx]
    prompt = np.random.choice(templates[is_has][template])

    return {
        'category': category,
        'group': group,
        'attribute': synonyms,
        'prompt': prompt.format(attr=', '.join(synonyms), noun=category, dobj=group)
    }


def conditional_distribution(attribute_idx, categories, label_vectors, classes_vector, category_id2idx, normalize=False):
    """Computes the distribution of classes for the given attribute

    Args:
        attribute_idx (int): index of the attribute to generate the prompt for
        categories (pd.DataFrame): dataframe with all categories and corresponding ids
        label_vectors (torch.Tensor): tensor of shape (#samples, #attributes) holding annotations for each sample
        classes_vector (torch.Tensor): tensor of shape (#samples) holding the class label for each sample
        category_id2idx (dict): mapping from a category id (COCO) to its index (OVAD)
        normalize (bool, optional): whether to normalize the distribution (i.e. sum to one). Defaults to False.

    Returns:
        torch.Tensor: distribution of classes for the given attribute
    """
    matches = (label_vectors == 1)[:, attribute_idx].nonzero().view(-1)
    bins, classes_vector_hist = classes_vector[matches].unique(return_counts=True)
    bins = torch.tensor([category_id2idx[b.item()] for b in bins])

    result = torch.zeros(len(categories), dtype=torch.long)
    if len(bins) == 0:
        return result
    result[bins] = classes_vector_hist    

    if normalize:
        result = result.float()
        result /= result.sum()

    return result


def generate_prompt_conditional(attribute_idx, idx2attr, idx2is_has, categories, templates, label_vectors, classes_vector, category_id2idx, on_empty_distribution='uniform', topk=None, topk_mode=None, template='none'):
    """Generate a prompt by binding the attribute to a conditionally sampled class

    Args:
        attribute_idx (int): index of the attribute to generate the prompt for
        idx2attr (dict): dictionary mapping an index to the textual attribute
        idx2is_has (dict): dictionary mapping an attribute (by its index) to its type, either 'is' or 'has'
        categories (pd.DataFrame): dataframe with all categories and corresponding ids
        templates (dictionary): templates to use to generate prompts
        label_vectors (torch.Tensor): tensor of shape (#samples, #attributes) holding annotations for each sample
        classes_vector (torch.Tensor): tensor of shape (#samples) holding the class label for each sample
        category_id2idx (dict): mapping from a category id (COCO) to its index (OVAD)
        on_empty_distribution (str, optional): default action if the conditional distribution does not exist or is invalid. Defaults to 'uniform'
        topk (int, optional): whether to consider only the top-k values in the distribution. Defaults to None
        topk_mode (str, optional): how to deal with top-k values when `topk` is set. It should be either `random` or `weighted`.
            `random` means that one value is randomly sampled among the top-k ones, regardless of their probabilities.
            `weighted` means that the top-k values are first re-weighted to sum up to 1 via a softmax operation, and then are sampled according
            to this probability distribution.
            Defaults to None.
        template (str, optional): which type of template to use. Defaults to 'none'

    Returns:
        dict: the bound category, the group of the attribute, the attribute itself, the prompt
    """
    
    assert label_vectors is not None, '`label_vectors` must not be None'
    assert classes_vector is not None, '`classes_vector` must not be None'
    assert category_id2idx is not None, '`category_id2idx` must not be None'
    assert topk_mode is not None or topk is None, '`topk_mode` must not be None if `topk` is set'
    
    # Extract attribute
    group, synonyms = idx2attr[str(attribute_idx)].split(':')
    synonyms = synonyms.split('/')

    # Get conditional distribution for attribute
    distribution_for_attribute = conditional_distribution(attribute_idx, categories, label_vectors, classes_vector, category_id2idx, normalize=True)
    # Edge case, which should happend only when there is no object in the current attribute's conditional distribution
    if distribution_for_attribute.sum().item() < 0.99:
        if on_empty_distribution == 'uniform':
            return generate_prompt_uniform(attribute_idx, idx2attr, idx2is_has, categories, templates, template=template)
        return None
    
    # Setting the sampled category to None so that we can sample in different ways without having
    # too many if-else conditions
    category = None

    if topk is not None:
        # Actual top-k
        if topk.is_integer():
            values, indices = distribution_for_attribute.topk(topk)
            indices = indices[values > 0]  # Filter only values that are > 0, as this could cause unintended behavior when sampling
        
        # Top-k with thresholding, i.e. k is determined dynamically
        else:
            values, indices = distribution_for_attribute.sort(descending=True)
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
        mask = torch.ones(len(distribution_for_attribute)) * float('-inf')
        mask[indices] = distribution_for_attribute[indices]
        distribution_for_attribute = mask

    if topk_mode == 'weighted':
        # Prepare the distribution for subsequent sampling
        distribution_for_attribute = distribution_for_attribute.softmax(dim=0)
    elif topk_mode == 'random':
        # Randomly sample a category
        indices = (~distribution_for_attribute.isinf()).nonzero().view(-1)
        category = categories.iloc[np.random.choice(indices.cpu())]['name']

    # Sample a category, based on conditional distribution
    # Sample only if a category has not been sampled yet --> this only happens when sampling top-k randomly, otherwise
    # a category has not been chosen yet at this point
    if category is None:
        category = categories.iloc[np.random.choice(np.arange(0, len(categories)), p=distribution_for_attribute.cpu().numpy())]['name']

    # Generate prompt
    is_has = idx2is_has[attribute_idx]
    prompt = np.random.choice(templates[is_has][template])

    return {
        'category': category,
        'group': group,
        'attribute': synonyms,
        'prompt': prompt.format(attr=', '.join(synonyms), noun=category, dobj=group)
    }


def get_negatives(attribute_idx, idx2group):
    """Get negative attributes (i.e. attributes in the same group as the given one)

    Args:
        attribute_idx (int, str): index of the positive attribute
        idx2group (torch.Tensor): mapping from attribute index to attribute group index

    Returns:
        torch.Tensor: indices of attributes in the same group as the given attribute
    """

    group = idx2group[int(attribute_idx)]
    attributes_in_group = (idx2group == group).nonzero().view(-1)
    
    # Remove the attribute itself from the group: it is not a negative!
    return attributes_in_group[attributes_in_group != int(attribute_idx)]


def generate_prompt_class_only(categories, classes_vector, distribution, templates, template='none'):
    """Generate a prompt by sampling the class/category without any binding to attributes

    Args:
        categories (pd.DataFrame): dataframe with all categories and corresponding ids
        classes_vector (torch.Tensor): tensor of shape (#samples) holding the class label for each sample
        distribution (str): distribution of classes
        templates (dictionary): templates to use to generate prompts
        template (str, optional): which type of template to use. Defaults to 'none'.

    Returns:
        dict: the bound category, the group of the attribute (always None), the attribute itself (always None), the prompt
    """

    assert distribution in ['uniform', 'test'], '`distribution` must be either "uniform" or "test"'
    
    # Generate prompt
    prompt = np.random.choice(templates['class_only'][template])

    if distribution == 'uniform':
        # Sample a category, randomly
        category = categories.iloc[np.random.choice(np.arange(0, len(categories)))]['name']
    elif distribution == 'test':
        _, counts = classes_vector.unique(return_counts=True)
        counts = counts.float() / counts.sum().float()
        category = categories.iloc[np.random.choice(np.arange(0, len(categories)), p=counts.cpu().numpy())]['name']

    return {
        'category': category,
        'group': None,
        'attribute': None,
        'prompt': prompt.format(noun=category)
    }


def generate_prompt_attribute_only(idx2attr, classes_vector, distribution, templates, template='none'):
    if distribution == 'uniform':
        pass
    elif distribution == 'test':
        pass

    return None


def generate_prompt(attribute_idx, idx2attr, idx2group, idx2is_has, categories, templates, get_negative_attributes=True, mode='uniform', template='none', label_vectors=None, classes_vector=None, category_id2idx=None, on_empty_distribution='uniform', attributes_vector=None):
    topk = None
    topk_mode = None
    if '_top-' in mode:
        mode, topk = mode.split('_top-')
        topk, topk_mode = topk.split('_')
        topk = float(topk)
    
    assert mode in [
        'uniform',
        'class_attribute_test',
        'class_only_uniform', 'class_only_test',
        'attribute_only_uniform', 'attribute_only_test',
        'llm_generated_uniform',
    ], 'mode is not valid'

    # Get negative prompts
    negatives = None
    negatives_prompt = None
    if get_negative_attributes:
        negatives_idx = get_negatives(attribute_idx, idx2group)
        negatives = [parse_attribute(negative_idx, idx2attr)[1] for negative_idx in negatives_idx]
        negatives_prompt = ', '.join(itertools.chain.from_iterable(negatives))
        
    if mode == 'uniform':
        result = generate_prompt_uniform(attribute_idx, idx2attr, idx2is_has, categories, templates, template=template)
    if mode == 'class_attribute_test':
        result = generate_prompt_conditional(attribute_idx, idx2attr, idx2is_has, categories, templates, label_vectors, classes_vector, category_id2idx, on_empty_distribution=on_empty_distribution, topk=topk, topk_mode=topk_mode, template=template)
    if 'class_only' in mode:
        result = generate_prompt_class_only(categories, classes_vector, mode.split('_')[-1], templates, template=template)
        # Force negative disabling, as the prompt should be about the class only
        get_negative_attributes = False
    if 'attribute_only' in mode:
        result = generate_prompt_attribute_only(mode.split('_')[-1])
    if mode == 'llm_generated_uniform':
        categories = pd.DataFrame.from_dict({'name': categories[str(attribute_idx)]})
        result = generate_prompt_uniform(attribute_idx, idx2attr, idx2is_has, categories, templates, template=template)

    if get_negative_attributes:
        result['negatives'] = negatives
        result['negatives_prompt'] = negatives_prompt
    return result


def generate_prompt_unbound(idx, class_or_attribute, categories, idx2attr, idx2group, idx2is_has, templates, get_negative=True, template='none'):
    assert class_or_attribute in ['class_only', 'category_only', 'attribute_only']

    category = None
    group = None
    attribute = None
    negatives = None
    negatives_prompt = None
    if class_or_attribute in ['class_only', 'category_only']:
        # Extract class
        category = categories.iloc[int(idx)]['name']

        # Generate prompt
        prompt = np.random.choice(templates['class_only'][template])
        prompt = prompt.format(noun=category)

        if get_negative:
            # Sorting just to avoid reproducibility errors
            negatives_idx = sorted(list(set(categories.index.tolist()).difference({int(idx)})))
            negatives = categories.iloc[negatives_idx]['name'].tolist()
            negatives_prompt = ', '.join(negatives)

    elif class_or_attribute in ['attribute_only']:
        # Extract attribute
        group, synonyms = idx2attr[str(idx)].split(':')
        synonyms = synonyms.split('/')
        attribute = synonyms

        # Generate prompt
        prompt = np.random.choice(templates['attribute_only'][idx2is_has[int(idx)]][template])
        prompt = prompt.format(attr=', '.join(synonyms), dobj=group)

        if get_negative:
            negatives_idx = get_negatives(idx, idx2group)
            negatives = [parse_attribute(negative_idx, idx2attr)[1] for negative_idx in negatives_idx]
            negatives_prompt = ', '.join(itertools.chain.from_iterable(negatives))

    return {
        'category': category,
        'group': group,
        'attribute': attribute,
        'prompt': prompt,
        'negatives': negatives,
        'negatives_prompt': negatives_prompt
    }
