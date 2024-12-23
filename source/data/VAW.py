from typing import Optional

from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import os

from ..utils import convert_bbox_format


class VAW(Dataset):
    def __init__(self,
                 images_path: str,
                 split: str = 'train',
                 load_images: bool = True,
                 cache_dir: Optional[str] = None) -> None:
        super().__init__()

        assert split in ['train', 'validation', 'test'], '`split` must be either "train", "validation", or "test"'

        self.images_path = images_path
        self.dataset = load_dataset('mikewang/vaw', trust_remote_code=True, cache_dir=cache_dir)[split]
        self.load_images = load_images

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        if self.load_images:
            image = Image.open(os.path.join(self.images_path, f"{sample['image_id']}.jpg"))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = image.crop(convert_bbox_format(sample['instance_bbox']))
        else:
            image = None

        return image, sample
    
    @staticmethod
    def get_collate_fn():
        return vaw_collate_fn


def vaw_collate_fn(batch):
    images = [x[0] for x in batch]
    metadata = [x[1] for x in batch]

    return images, metadata
