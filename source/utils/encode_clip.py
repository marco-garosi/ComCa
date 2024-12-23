import torch
import torch.nn.functional as F
from PIL import Image
import itertools


def clip_encode_images(model, processor, images, model_from='OpenCLIP', device='cpu', normalize=False):
    assert isinstance(images, list) or isinstance(images, torch.Tensor), '`images` must be a `torch.Tensor` or a list of `str` or `PIL.Image`'
    assert len(images) > 0, '`images` must contain at least one element'
    assert model_from in ['OpenCLIP', 'open_clip', 'HF', 'hf_clip', 'BLIP', 'X_VLM'], '`model_from` must be either OpenCLIP, open_clip, HF, hf_clip (HuggingFace), BLIP, X_VLM'
    
    if isinstance(images[0], str):
        images = [Image.open(file) for file in images]
    
    # OpenCLIP model
    if model_from in ['OpenCLIP', 'open_clip']:
        if isinstance(images[0], Image.Image):
            inputs = torch.stack([processor(image) for image in images]).to(device)
        else:
            inputs = images.to(device)

        with torch.no_grad():
            features = model.encode_image(inputs)

    # HuggingFace model
    elif model_from in ['HF', 'hf_clip']:
        if isinstance(images[0], Image.Image):
            images = [image if image.mode == 'RGB' else image.convert('RGB') for image in images]
            inputs = processor(images=images, return_tensors='pt').to(device)
        else:
            inputs = {'pixel_values': images.to(device)}

        with torch.no_grad():
            if hasattr(model, 'get_image_features'):
                features = model.get_image_features(**inputs)
            else:
                features = model.vision_model(**inputs).last_hidden_state[:, 0, :]

    elif model_from in ['BLIP']:
        if isinstance(images[0], Image.Image):
            inputs = torch.stack([processor(image) for image in images]).to(device)
        else:
            inputs = images.to(device)
            
        with torch.no_grad():
            embeddings = model.visual_encoder(images.to(device))
            features = F.normalize(
                model.vision_proj(embeddings[:, 0, :]), dim=-1
            )
            normalize = False

    elif model_from in ['X_VLM']:
        if isinstance(images[0], Image.Image):
            inputs = torch.stack([processor(image) for image in images]).to(device)
        else:
            inputs = images.to(device)

        with torch.no_grad():
            embeddings = model.vision_encoder(images.to(device))
            features = F.normalize(
                model.vision_proj(embeddings[:, 0, :]), dim=-1
            )
            normalize = False

    # Should never happen
    else:
        raise RuntimeError('`model_from` is not valid')
    
    if normalize:
        features /= features.norm(dim=-1, keepdim=True)

    return features


def clip_encode_text(text_list, model, tokenizer, device='cpu', model_from='OpenCLIP', normalize=False, padding: bool | str=True):
    assert model_from in ['OpenCLIP', 'open_clip', 'HF', 'hf_clip', 'BLIP', 'X_VLM'], '`model_from` must be either OpenCLIP, open_clip, HF, hf_clip (HuggingFace), BLIP, X_VLM'

    if isinstance(text_list[0], list):
        # it is a list of list of strings
        avg_synonyms = True
        sentences = list(itertools.chain.from_iterable(text_list))
    elif isinstance(text_list[0], str):
        avg_synonyms = False
        sentences = text_list
    
    # OpenCLIP model
    if model_from in ['OpenCLIP', 'open_clip']:
        text = tokenizer(sentences).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
    
    # HuggingFace model
    elif model_from in ['HF', 'hf_clip']:
        text = tokenizer(text=sentences, return_tensors='pt', padding=padding).to(device)
        with torch.no_grad():
            if hasattr(model, 'get_text_features'):
                text_features = model.get_text_features(**text)
            else:  # e.g. BLIP
                text_features = model.text_encoder(**text).last_hidden_state[:, 0, :]

    elif model_from in ['BLIP']:
        text = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            text_output = model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode='text',
            )
            text_features = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
            normalize = False  # already normalized here!

    elif model_from in ['X_VLM']:
        text = tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=30,
            return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            text_output = model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                mode='text'
            )
            text_features = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
            normalize = False  # already normalized here!

    # Should never happen
    else:
        raise RuntimeError('`model_from` is not valid')
    
    if normalize:
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Average synonyms
    # This works if the input `text_list` is a list of lists of strings,
    # where each sublist contains the synonyms of a given attribute
    if avg_synonyms:
        synonyms_per_cat = [len(x) for x in text_list]
        text_features = text_features.split(synonyms_per_cat, dim=0)
        text_features = [x.mean(dim=0) for x in text_features]
        text_features = torch.stack(text_features, dim=0)

    return text_features
