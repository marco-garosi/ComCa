from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import clip_encode_images


class Predictor(nn.Module):
    def __init__(self,
                 backbone: Callable,
                 backbone_from: str,
                 backbone_processor,
                 backbone_device: torch.device,
                 text_features: Optional[torch.Tensor],
                 len_synonyms: int,
                 average_syn: bool = False,
                 batch_size: bool = 1,
                 sigmoid: bool = False,
                 model_arch: str = '',
                 **kwargs
                ):
        super().__init__()

        self.backbone = backbone
        self.backbone_from = backbone_from
        self.backbone_processor = backbone_processor
        self.backbone_from = backbone_from
        self.backbone_device = backbone_device
        
        self.text_features = text_features
        self.len_synonyms = len_synonyms
        self.average_syn = average_syn
        self.batch_size = batch_size

        self.sigmoid = sigmoid
        self.model_arch = model_arch

    def aggregate_synonyms(self, logits, single_split: bool = False):
        # Split into synonyms
        if single_split:
            len_synonyms = torch.ones(len(self.len_synonyms)).int().tolist()
        else:
            len_synonyms = self.len_synonyms
        x_attrs_syn = logits.split(len_synonyms, dim=1)

        # Aggregate synonyms to just only logit
        # --> If an attribute has 5 synonyms, there will be 5 prompts for it, which in turn
        #       generate 5 logit vectors. We want just a single logit vector for a given attribute,
        #       so we either take the one with the highest score, or we average them, based on
        #       the command line argument `average_syn`
        x_attrs_maxsyn = []
        x_attrs_idxsyn = []
        for x_syn in x_attrs_syn:
            if self.average_syn:
                xmax_val = x_syn.mean(axis=1)
                xmax_idx = torch.zeros((1, self.batch_size))
            else:
                xmax_val, xmax_idx = x_syn.max(axis=1)
            x_attrs_maxsyn.append(xmax_val)
            x_attrs_idxsyn.append(xmax_idx)
        
        idx_attrs = torch.stack(x_attrs_idxsyn, axis=1)
        x_attrs = torch.stack(x_attrs_maxsyn, axis=1)

        return idx_attrs, x_attrs

    def forward(self,
                images,
                scale_base_logits: float = 1.0,
                image_features: Optional[torch.Tensor] = None,
                text_features: Optional[torch.Tensor] = None,
                **kwargs
                ):
        if image_features is None:
            image_features = clip_encode_images(self.backbone, self.backbone_processor, images, model_from=self.backbone_from, device=self.backbone_device, normalize=True)

        text_features = self.text_features if text_features is None else text_features
        early_split_occurred = False

        # SigLIP family of models
        if any(arch in self.model_arch for arch in ['siglip']):
            cache = kwargs.get('cache')
            if cache is not None:
                # Option 1: aggregate text features
                text_features = text_features.split(self.len_synonyms, dim=0)
                text_features = torch.stack([
                    x.mean(dim=0)
                    for x in text_features
                ])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                base_logits = (
                    scale_base_logits
                    * torch.matmul(text_features, image_features.t()).t()
                )

                cache_logits = cache(base_logits, image_features, None, backbone_logit_scale=self.backbone.logit_scale.exp(), backbone_logit_bias=self.backbone.logit_bias, activation='sigmoid').t()
                cache_logits *= self.backbone.logit_scale.exp()
                logits_per_text = cache_logits + self.backbone.logit_bias
                logits_per_text = F.softmax(logits_per_text.T).T

            else:
                logits_per_text = (
                    scale_base_logits
                    * torch.matmul(text_features, image_features.t())
                    * self.backbone.logit_scale.exp()
                    + self.backbone.logit_bias
                )
                logits_per_text = F.sigmoid(logits_per_text)

            logits = logits_per_text.t()
        
        # Models other than SigLIP
        else:
            logits = scale_base_logits * image_features @ text_features.T

        if early_split_occurred:
            x_attrs = logits
        else:
            idx_attrs, x_attrs = self.aggregate_synonyms(logits, single_split = len(self.len_synonyms) == logits.shape[-1])

        return {
            'image_features': image_features,
            'scores': x_attrs,
            'idx_attrs': idx_attrs,
        }
