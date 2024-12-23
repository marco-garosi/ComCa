import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .RetrievalDatabase import RetrievalDatabase

__all__ = ["RetrievalVocabulary"]


class BaseVocabulary(ABC, torch.nn.Module):
    """Base class for a vocabulary for image classification."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> list[list[str]]:
        values = self.forward(*args, **kwargs)
        return values

    @abstractmethod
    def forward(self, *args, **kwargs) -> list[list[str]]:
        """Forward pass."""
        raise NotImplementedError


class RetrievalVocabulary(BaseVocabulary):
    """Vocabulary based on captions from an external database.

    Args:
        database_name (str): Name of the database to use.
        databases_dict_fp (str): Path to the databases dictionary file.
        num_samples (int): Number of samples to return. Default is 40.
    """

    def __init__(
        self, *args, database: RetrievalDatabase, num_samples: int = 10, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

        self.database = database

    def __call__(self, *args, **kwargs) -> list[list[str]]:
        values = super().__call__(*args, **kwargs)

        # keep only the `num_samples` first words
        num_samples = self.num_samples
        values = [value[:num_samples] for value in values]

        return values

    def forward(self, *args, images_z: Optional[torch.Tensor] = None, **kwargs) -> list[list[str]]:
        """Create a vocabulary for a batch of images.

        Args:
            images_z (torch.Tensor): Batch of image embeddings.
        """
        assert images_z is not None

        images_z = images_z / images_z.norm(dim=-1, keepdim=True)
        images_z = images_z.cpu().detach().numpy().tolist()

        if isinstance(images_z[0], float):
            images_z = [images_z]

        query = np.matrix(images_z).astype("float32")
        results = self.database.query(query, modality="text", num_samples=self.num_samples)
        vocabularies = [[r["caption"] for r in result] for result in results]

        return vocabularies
