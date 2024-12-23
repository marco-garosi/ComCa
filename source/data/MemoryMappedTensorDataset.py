import torch
import numpy as np
from torch.utils.data import Dataset


class MemoryMappedTensorDataset(Dataset):
    def __init__(self, filename, shape=None, dtype=np.float32):
        if filename.endswith('.npy') and shape is None:
            self._memmap = np.load(filename, mmap_mode='r')
        else:
            self._memmap = np.memmap(filename, dtype=dtype, mode='r', shape=shape)

    @property
    def shape(self):
        return self._memmap.shape

    def __len__(self):
        return self._memmap.shape[0]
    
    def __getitem__(self, idx):
        return torch.from_numpy(self._memmap[idx].copy())
