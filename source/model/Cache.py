from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cache(nn.Module):
    def __init__(self,
                 alpha: float,
                 beta: float,
                 keys: torch.Tensor,
                 values: torch.Tensor,
                 scale_logits: str = 'none',
                 softmax: str = 'none',
                ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.keys = keys
        self.values = values
        
        self.scale_logits = scale_logits
        self.softmax = softmax

    def forward(self, base_logits, image_features, obj_label: Optional[torch.Tensor] = None, **kwargs):
        scale_logits = self.scale_logits
        values = self.values

        # Cache specific for class/category
        if isinstance(self.keys, dict):
            cache_logits = []
            for idx, image_feature in enumerate(image_features):
                affinity = image_feature.unsqueeze(0) @ self.keys[obj_label[idx].item()]
                cache_logits.append(
                    ((-1) * (self.beta - self.beta * affinity)).exp() @ self.values[obj_label[idx].item()]
                )

            # Squeezing
            # Since `cache_logits` is constructed by appending tensors with shape (1, ...), stacking them
            # produces a tensor with shape (bs, 1, ...)
            # While concatenating would solve the issue, stacking is more intuitive when reading this code
            # But we need to remove that extra dimension, so we squeeze it
            cache_logits = torch.stack(cache_logits).squeeze(1)
        
        # Single cache
        else:
            if '_sl_according_to_distribution' in scale_logits:
                scale_logits = scale_logits.replace('_sl_according_to_distribution', '')

                values = torch.clone(self.values)
                mean = values.mean(dim=-1, keepdim=True)
                std = values.std(dim=-1, keepdim=True)
                target_mean = base_logits.mean()
                target_std = base_logits.std()

                values = target_mean + (values - mean) * (target_std / std)

            # Plain Tip-Adapter
            affinity = image_features @ self.keys

            activation = kwargs.get('activation', 'default')
            if activation == 'sigmoid':
                cache_logits = torch.sigmoid(affinity) @ values
            else:
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ values
        
        # Here, `cache_logits` has the right shape regardless how it's been computed
        if '_cl_minmax' in scale_logits:
            scale_logits = scale_logits.replace('_cl_minmax', '')

            cache_logits -= cache_logits.min(dim=-1, keepdim=True).values
            cache_logits /= cache_logits.max(dim=-1, keepdim=True).values
        elif '_cl_max' in scale_logits:
            scale_logits = scale_logits.replace('_cl_max', '')

            cache_logits /= cache_logits.max(dim=-1, keepdim=True).values
        elif '_cl_normalize_wrt_base' in scale_logits:
            scale_logits = scale_logits.replace('_cl_normalize_wrt_base', '')

            cache_logits -= cache_logits.min(dim=-1, keepdim=True).values
            cache_logits /= cache_logits.max(dim=-1, keepdim=True).values

            cache_logits -= base_logits.mean(dim=-1, keepdim=True)
            cache_logits /= base_logits.std(dim=-1, keepdim=True)
        
        tip_logits = base_logits + cache_logits * self.alpha

        # Scaling
        if scale_logits == 'minmax':
            final_logits = tip_logits - tip_logits.min(dim=-1, keepdim=True).values
            final_logits /= final_logits.max(dim=-1, keepdim=True).values
        
        elif scale_logits == 'max':
            final_logits = tip_logits / tip_logits.max(dim=-1, keepdim=True).values
        
        else:
            final_logits = tip_logits

        # Softmax
        if self.softmax == 'overall':
            final_logits = F.softmax(final_logits, dim=-1)

        return final_logits
