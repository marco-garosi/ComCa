from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import json
import os
import copy

from ..utils import convert_bbox_format


class VAWDetection(Dataset):
    def __init__(self,
                 annotations_path: str,
                 images_path: str,
                 split: str = 'train',
                 load_images: bool = True,
                 cache_dir: Optional[str] = None) -> None:
        super().__init__()

        assert split in ['train', 'validation', 'test'], '`split` must be either "train", "validation", or "test"'

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        self.images_path = images_path
        self.dataset = load_dataset('mikewang/vaw', trust_remote_code=True, cache_dir=cache_dir)[split]
        self.load_images = load_images

    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_path, self.annotations['images'][idx]['file_name']))
        image_id = self.annotations['images'][idx]['id']
        annotations = [x for x in self.annotations['annotations'] if x['image_id'] == image_id]

        bounding_boxes = [convert_bbox_format(copy.deepcopy(x['bbox'])) for x in annotations]
        att_vectors = torch.tensor([x['att_vec'] for x in annotations])

        return image, {
            'raw': annotations,
            'image_id': image_id,
            'bounding_boxes': bounding_boxes,
            'attribute_vectors': att_vectors,
        }
    
    @staticmethod
    def get_collate_fn():
        return vaw_collate_fn


def vaw_collate_fn(batch):
    images = [x[0] for x in batch]
    metadata = [x[1] for x in batch]

    return images, metadata
