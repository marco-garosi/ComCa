from typing import Optional

import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import glob
import copy

from ..utils import convert_bbox_format

class OVADDetection(Dataset):
    def __init__(
        self,
        annotations_path: str,
        images_path: str,
        split: str = 'validation',
        transform = None,
    ):
        super().__init__()

        assert split in ['test'], '`split` must be "test"'

        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.images_path = images_path
        # if split == 'train':
        #     self.folder = 'train2017'
        # elif split == 'validation':
        #     self.folder = 'val2017'
        if split == 'test':
            self.folder = 'val2017'
        self.path = os.path.join(self.images_path, self.folder)

        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path, self.annotations['images'][idx]['file_name']))
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
        return ovaddetection_collate_fn
    
def ovaddetection_collate_fn(batch):
    images = [x[0] for x in batch]
    metadata = [x[1] for x in batch]

    return images, metadata
