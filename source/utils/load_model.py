def load_model(model_from, model_arch, pretrained=None, device='cpu', cache_dir=None, image_size=-1):
    """Load the model specified by `model_arch` from the hub specified via `model_from`
    """

    assert model_from in ['OpenCLIP', 'open_clip', 'HF', 'hf_clip', 'BLIP', 'X_VLM'], '`model_from` must be either OpenCLIP, open_clip, HF, hf_clip (HuggingFace), BLIP, X_VLM'

    if model_from in ['OpenCLIP', 'open_clip']:
        import open_clip

        model, _, processor = open_clip.create_model_and_transforms(model_arch, pretrained=pretrained, cache_dir=cache_dir)
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(model_arch)
    
    elif model_from in ['HF', 'hf_clip']:
        if 'blip' in model_arch.lower():
            from transformers import BlipProcessor, BlipForImageTextRetrieval, BlipModel
            
            model = BlipForImageTextRetrieval.from_pretrained(model_arch).to(device)
            # model = BlipModel.from_pretrained(model_arch).to(device)
            processor = BlipProcessor.from_pretrained(model_arch)
            tokenizer = processor

        elif 'clip' in model_arch.lower():
            from transformers import CLIPProcessor, CLIPModel

            model = CLIPModel.from_pretrained(model_arch).to(device)
            processor = CLIPProcessor.from_pretrained(model_arch)
            tokenizer = processor

        else:
            from transformers import AutoProcessor, AutoModel

            model = AutoModel.from_pretrained(model_arch).to(device)
            processor = AutoProcessor.from_pretrained(model_arch)
            tokenizer = processor

    elif model_from == 'BLIP':
        model = load_blip(model_arch, image_size=image_size, device=device)
        tokenizer = model.tokenizer
        processor = model.tokenizer

    elif model_from == 'X_VLM':
        model, tokenizer = load_xvlm(model_arch, device=device)       
        processor = tokenizer

    else:
        from all_clip import load_clip

        pretrained = f"/{pretrained}" if len(pretrained) > 0 else ""
        model, processor, tokenizer = load_clip(f"{model_from}:{model_arch}{pretrained}", device=device, use_jit=False)

    return model, processor, tokenizer


def load_xvlm(model_arch, device='cpu'):
    import sys
    sys.path.append('./source/backbones/X_VLM')

    import yaml
    import torch
    from source.backbones.X_VLM.models.model_retrieval import XVLM
    from source.backbones.X_VLM.models.tokenization_bert import BertTokenizer
    from source.backbones.X_VLM.models.tokenization_roberta import RobertaTokenizer

    # From OVAD
    model_versions = {
    "pretrained16M": "source/backbones/X_VLM/configs/Pretrain_XVLM_base_16m.yaml",
    "pretrained4M": "source/backbones/X_VLM/configs/Pretrain_XVLM_base_4m.yaml",
    "pretrained16M2": "source/backbones/X_VLM/configs/Pretrain_XVLM_base_16m.yaml",
    "pretrained4M2": "source/backbones/X_VLM/configs/Pretrain_XVLM_base_4m.yaml",
    "cocoRetrieval": "source/backbones/X_VLM/configs/Retrieval_coco.yaml",
    }
    model_weights = {
        "pretrained16M": "source/backbones/X_VLM/weights/16m_base_model_state_step_199999.th",
        "pretrained4M": "source/backbones/X_VLM/weights/4m_base_model_state_step_199999.th",
        "pretrained16M2": "source/backbones/X_VLM/weights/16m_base_model_state_step_199999.th",
        "pretrained4M2": "source/backbones/X_VLM/weights/4m_base_model_state_step_199999.th",
        "pt16m_cocoRetrieval": "source/backbones/X_VLM/weights/itr_coco/checkpoint_9.pth",
        "pt16m_flickrRetrieval": "source/backbones/X_VLM/weights/itr_flickr/checkpoint_best.pth",
        "pt4m_cocoRetrieval": "source/backbones/X_VLM/weights/4m_base_finetune/itr_coco/checkpoint_best.pth",
        "pt4m_flickrRetrieval": "source/backbones/X_VLM/weights/4m_base_finetune/itr_flickr/checkpoint_best.pth",
        "pt16m_cocoRef": "source/backbones/X_VLM/weights/refcoco/checkpoint_best.pth",
        "pt16m_cocoRefBox": "source/backbones/X_VLM/weights/refcoco_bbox/checkpoint_best.pth",
        "pt4m_cocoRef": "source/backbones/X_VLM/weights/4m_base_finetune/refcoco/checkpoint_best.pth",
        "pt4m_cocoRefBox": "source/backbones/X_VLM/weights/4m_base_finetune/refcoco_bbox/checkpoint_best.pth",
    }

    if model_arch in model_versions.keys():
        config = model_versions[model_arch]
    else:
        config = model_versions['cocoRetrieval']
    config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    config['vision_config'] = 'source/backbones/X_VLM/' + config['vision_config']
    config['text_config'] = 'source/backbones/X_VLM/' + config['text_config']
    model = XVLM(config=config)

    weights = model_weights[model_arch]
    checkpoint = torch.load(weights, map_location=device)
    if "model" in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for key, val in state_dict.items():
        if '.bert.' in key:
            new_key = key.replace('.bert.', '.')
            new_state_dict[new_key] = val
        else:
            new_state_dict[key] = val

    msg = model.load_state_dict(new_state_dict, strict=False)

    text_encoder_name = config['text_encoder'].split('/')[-1]
    if config.get('use_roberta', False):
        tokenizer = RobertaTokenizer.from_pretrained(text_encoder_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(text_encoder_name)

    return model.to(device), tokenizer


def load_blip(model_arch, image_size=-1, device='cpu'):
    import yaml
    import sys
    sys.path.append('./source/backbones/BLIP')
    
    from source.backbones.BLIP.models.blip import load_checkpoint

    # From OVAD
    model_versions = {
        'base': 'source/backbones/BLIP/weights/model_base.pth',
        'base14M': 'source/backbones/BLIP/weights/model_base_14M.pth',
        'baseCapfilt': 'source/backbones/BLIP/weights/model_base_capfilt_large.pth',
        'baseRetrieval': 'source/backbones/BLIP/weights/model_base_retrieval_coco.pth',
        'baseVQA': 'source/backbones/BLIP/weights/model_vqa.pth',
        'baseRetrievalFlicker': 'source/backbones/BLIP/weights/model_base_retrieval_flickr.pth',
    }
    
    if model_arch == 'baseRetrieval':
        from source.backbones.BLIP.models.blip_itm import blip_itm

        config = yaml.load(
            open('source/backbones/BLIP/configs/retrieval_coco.yaml', 'r'), Loader=yaml.Loader
        )
        med_config = 'source/backbones/BLIP/configs/med_config.json'
        model = blip_itm(
            med_config=med_config,
            pretrained=model_versions[model_arch],
            image_size=config['image_size'],
            vit='base',
        )
    elif model_arch == 'baseRetrievalFlicker':
        from source.backbones.BLIP.models.blip_itm import blip_itm

        config = yaml.load(
            open('source/backbones/BLIP/configs/retrieval_flickr.yaml', 'r'), Loader=yaml.Loader
        )
        med_config = 'source/backbones/BLIP/configs/med_config.json'
        model = blip_itm(
            med_config=med_config,
            pretrained=model_versions[model_arch],
            image_size=config['image_size'],
            vit='base',
        )
    else:
        config = yaml.load(
            open('source/backbones/BLIP/configs/pretrain.yaml', 'r'), Loader=yaml.Loader
        )
        med_config = 'source/backbones/BLIP/configs/bert_config.json'
        if image_size != -1 and image_size != config['image_size']:
            image_size = image_size

            from source.backbones.BLIP.models.blip_itm import blip_itm

            model = blip_itm(
                med_config=med_config,
                pretrained='',
                image_size=config['image_size'],
                vit='base',
            )
            weights = model_versions[model_arch]
            model, msg = load_checkpoint(model, weights)
            print(msg)
        else:
            image_size = config['image_size']

            from source.backbones.BLIP.models.blip_pretrain import blip_pretrain as blip

            model = blip(
                med_config=med_config,
                image_size=image_size,
                vit=config['vit'],
                vit_grad_ckpt=config['vit_grad_ckpt'],
                vit_ckpt_layer=config['vit_ckpt_layer'],
                queue_size=config['queue_size'],
            )
            weights = model_versions[model_arch]
            model, msg = load_checkpoint(model, weights)
            print(msg)

    return model.to(device)
