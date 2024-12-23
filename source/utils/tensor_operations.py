import torch


def intersect_tensors(a: torch.Tensor, b: torch.Tensor):
    return torch.tensor([
        x for x in
        set(a.unique().cpu().numpy()).intersection(set(b.unique().cpu().numpy()))
    ], device=a.device)


def difference_tensors(a: torch.Tensor, b: torch.Tensor):
    return torch.tensor([
        x for x in
        set(a.unique().cpu().numpy()).difference(set(b.unique().cpu().numpy()))
    ], device=a.device)
