from torch.utils.data import DataLoader

from . import *


def get_dataloader(dataset: str, **kwargs):
    if dataset == 'OVAD':
        return get_ovad_dataloader(
            kwargs['data_root'],
            kwargs.get('image_size', 224),
            kwargs.get('batch_size', 1),
            kwargs.get('num_workers', 1),
            kwargs.get('load_images', True),
            kwargs.get('return_raw_images', False)
        )

    if dataset == 'OVAD_DET':
        dataset = OVADDetection(
            kwargs['annotations_path'],
            kwargs['images_path'],
            kwargs.get('split', 'train'),
        )

        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 1),
            shuffle=kwargs.get('shuffle', False),
            num_workers=kwargs.get('num_workers', 1),
            collate_fn=OVADDetection.get_collate_fn()
        )
    
    if dataset == 'VAW':
        dataset = VAW(
            kwargs['images_path'],
            kwargs.get('split', 'train'),
            load_images=kwargs.get('load_images', True),
            cache_dir=kwargs.get('cache_dir', None)
        )
        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 1),
            shuffle=kwargs.get('shuffle', False),
            num_workers=kwargs.get('num_workers', 1),
            collate_fn=VAW.get_collate_fn()
        )
    
    if dataset == 'VAW_DET':
        dataset = VAWDetection(
            kwargs['annotations_path'],
            kwargs['images_path'],
            kwargs.get('split', 'train'),
            load_images=kwargs.get('load_images', True),
            cache_dir=kwargs.get('cache_dir', None)
        )
        return DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 1),
            shuffle=kwargs.get('shuffle', False),
            num_workers=kwargs.get('num_workers', 1),
            collate_fn=VAW.get_collate_fn()
        )
