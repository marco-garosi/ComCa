import json
import glob
import os
from diffusers.utils import load_image


def load_metadata(folder, base_dir=''):
    metadata_store = []
    epochs = glob.glob(os.path.join(base_dir, folder, f'metadata_store_epoch_*.json'))
    epochs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for epoch in epochs:
        with open(epoch, 'r') as f:
            metadata_store.append(json.load(f))

    return metadata_store


def load_images(epoch, folder, base_dir=''):
    images_store = {}
    images_paths = glob.glob(os.path.join(base_dir, folder, '*', f'epoch_{epoch}.png'))
    images_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for image_path in images_paths:
        attr_id = os.path.dirname(os.path.normpath(image_path)).split('/')[-1]
        images_store[attr_id] = load_image(image_path)

    return images_store
