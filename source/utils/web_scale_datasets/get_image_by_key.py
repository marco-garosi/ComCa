from PIL import Image
import os
import io
import tarfile
from tqdm import tqdm


def get_image_by_key(key, root):
    tar_key = key[:5]
    corresponding_tar_file = os.path.join(root, f"{tar_key}.tar")
    tar = tarfile.open(corresponding_tar_file)
    image_data = tar.extractfile(f"{key}.jpg").read()
    image_original = Image.open(io.BytesIO(image_data)).convert('RGB')

    return image_original


def get_images_by_key(keys, root):
    # Group images by tar to perform batch read

    img2tar = {}

    for x in keys:
        prefix = x[:5]
        if prefix not in img2tar:
            img2tar[prefix] = []
        
        img2tar[prefix].append(x)

    # Iterate over tar files
    all_images = {}
    for tar_key, images in tqdm(img2tar.items()):
        corresponding_tar_file = os.path.join(root, f"{tar_key}.tar")
        tar = tarfile.open(corresponding_tar_file)

        for key in images:
            image_data = tar.extractfile(f"{key}.jpg").read()
            image_original = Image.open(io.BytesIO(image_data)).convert('RGB')
            all_images[key] = image_original

    return all_images
