import dill as pickle
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CocoDetection
import numpy as np
import pickle
import random
from PIL import ImageOps, ImageFilter
import json


class OVAD_Boxes(Dataset):
    def __init__(self, root, transform=None, train=False, fold=-1, load_images=True, return_raw_image=False):
        with open(os.path.join(root, "bb_labels", "ovad_labels.pkl"), "rb") as f:
            obj_att_labels = pickle.load(f)

        self.box_index = obj_att_labels["box_index"]
        self.img_ins_files = obj_att_labels["img_ins_files"]
        self.obj_labels = obj_att_labels["obj_labels"]
        self.att_labels = obj_att_labels["att_labels"]
        self.load_images = load_images
        self.return_raw_image = return_raw_image

        if fold in {1, 2, 3, 4}:
            with open(os.path.join(root, "bb_labels", "ovad_4fold_ids.pkl"), "rb") as f:
                box_index = pickle.load(f)
            if train:
                img_ids = box_index["train_img_ids"][fold]
                box_ids = box_index["train_box_ids"][fold]
            else:
                img_ids = box_index["val_img_ids"][fold]
                box_ids = box_index["val_box_ids"][fold]

            self.img_ids = img_ids
            self.box_ids = box_ids
        else:
            img_ids = [
                int(os.path.basename(file_name).split("_")[0])
                for file_name in self.img_ins_files
            ]
            img_ids = list(set(img_ids))
            img_ids.sort()
            self.img_ids = img_ids
            self.box_ids = self.box_index

        self.train = train
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        box_id = self.box_ids[index]
        if self.load_images:
            img_path = self.img_ins_files[box_id]
            img_path = os.path.join(self.root, 'bb_images', os.path.basename(img_path))
            img_raw = Image.open(img_path)
            img_raw = img_raw.convert('RGB')
            img = self.transform(img_raw)
        else:
            img = torch.empty(1)

        att_label = self.att_labels[box_id]
        # att_label = torch.FloatTensor(att_label)
        obj_label = self.obj_labels[box_id]
        # obj_label = torch.FloatTensor(obj_label)

        if self.return_raw_image:
            return img_raw, img, (att_label, obj_label)
        else:
            return img, (att_label, obj_label)
        

    def __len__(self):
        return len(self.box_ids)
