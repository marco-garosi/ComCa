from typing import Optional, Callable

import torch

from .Predictor import Predictor
from .Cache import Cache


class PredictorWithCache(Predictor):
    def __init__(self,
                 backbone: Callable,
                 backbone_from: str,
                 backbone_processor,
                 backbone_device: torch.device,
                 text_features: torch.Tensor,
                 len_synonyms: int,
                 cache: Cache,
                 average_syn: bool = False,
                 sigmoid: bool = False,
                 model_arch: str = '',
                 **kwargs
                ):
        super().__init__(
                         backbone,
                         backbone_from,
                         backbone_processor,
                         backbone_device,
                         text_features,
                         len_synonyms,
                         average_syn=average_syn,
                         sigmoid=sigmoid,
                         model_arch=model_arch,
                         )
        
        self.cache = cache

    def forward(self,
                images,
                scale_base_logits: float = 1.0,
                obj_label: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                **kwargs
                ):
        # Base model logits
        out = super().forward(images, scale_base_logits=scale_base_logits, image_features=image_features, cache=self.cache)
        image_features = out['image_features']
        x_attrs = out['scores']
        idx_attrs = out['idx_attrs']

        # Cache
        if not any(arch in self.model_arch for arch in ['siglip']):
            x_attrs = self.cache(x_attrs, image_features, obj_label)

        return {
            'image_features': image_features,
            'scores': x_attrs,
            'idx_attrs': idx_attrs,
        }
