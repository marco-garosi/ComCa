import torch

def get_device(allow_accelerator=True):
    """Get the best computing device available

    If `allow_accelerator` is False, this always returns 'cpu'.
    Otherwise, if CUDA is available, it returns 'cuda'.
    If CUDA is not available, it searches for MPS (Metal Performance Shader, available on Apple devices).
    If MPS is not available, it returns 'cpu', because both CUDA and MPS failed. Otherwise, it returns 'mps'.

    Args:
        allow_accelerator (bool, optional): whether to allow the use of accelerators, such as GPUs. Defaults to True.

    Returns:
        torch.device: device to perform calculations on
    """

    if not allow_accelerator:
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        return torch.device('cuda')

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
            
        return 'cpu'

    else:
        return torch.device('mps')
