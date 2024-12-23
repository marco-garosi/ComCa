from typing import Optional

from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not os.path.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item['label'],
            # 'domain': item.domain,
            'impath': item['impath']
        }

        img0 = read_image(item['impath'])

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_cache_model_with_features(cfg, train_loader_cache: Optional[DataLoader] = None, clip_model=None, data_is_normalized=False, is_values_to_encode=True, save_cache: bool = True, print_output: bool = True, device='cpu'):
    path = '{prefix}_{t}__{model}__{shots}_shots{suffix}.pt'
    if cfg['suffix'] is not None and len(cfg['suffix']) > 0:
        suffix = '_' + cfg['suffix']
    else:
        suffix = ''
    
    if cfg['load_cache'] == False:
        if train_loader_cache is None:
            raise Exception('`train_loader_cache` must not be None')
        
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                if print_output:
                    print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                    pbar = lambda x: tqdm(x)
                else:
                    pbar = lambda x: x
                for i, (images, target) in enumerate(pbar(train_loader_cache)):
                    if cfg['encode_images']:
                        images = images.to(device)
                        image_features = clip_model.encode_image(images)
                    else:
                        image_features = images
                    
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(device)
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
 
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)

        if not data_is_normalized:
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        
        if is_values_to_encode:
            cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        else:
            cache_values = torch.cat(cache_values, dim=0)

        if save_cache:
            os.makedirs(cfg['cache_dir'], exist_ok=True)
            torch.save(cache_keys, os.path.join(cfg['cache_dir'], path.format(prefix=cfg['prefix'], t='keys', model=cfg['model_name_slug'], shots=str(cfg['shots']), suffix=suffix)))
            torch.save(cache_values, os.path.join(cfg['cache_dir'], path.format(prefix=cfg['prefix'], t='values', model=cfg['model_name_slug'], shots=str(cfg['shots']), suffix=suffix)))

    else:
        cache_keys = torch.load(os.path.join(cfg['cache_dir'], path.format(prefix=cfg['prefix'], t='keys', model=cfg['model_name_slug'], shots=str(cfg['shots']), suffix=suffix))).to(device)
        cache_values = torch.load(os.path.join(cfg['cache_dir'], path.format(prefix=cfg['prefix'], t='values', model=cfg['model_name_slug'], shots=str(cfg['shots']), suffix=suffix))).to(device)

    if print_output:
        print(os.path.join(cfg['cache_dir'], path.format(prefix=cfg['prefix'], t='keys', model=cfg['model_name_slug'], shots=str(cfg['shots']), suffix=suffix)))

    return cache_keys, cache_values
