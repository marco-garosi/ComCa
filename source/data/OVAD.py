import torch
from torchvision.transforms import transforms
from ovad.data_loader import OVAD_Boxes


def get_ovad_dataloader(data_root, image_size: int, batch_size: int, num_workers: int = 1, load_images: bool = True, return_raw_images: bool = False):
    channel_stats = dict(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    
    transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats),
        ]
    )

    dataset = OVAD_Boxes(root=data_root, transform=transform, load_images=load_images, return_raw_image=return_raw_images)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn_ovad_dataloader if return_raw_images else None
    )

    return data_loader


# Define collation function for dataset returning raw images (i.e. PIL Image)
def collate_fn_ovad_dataloader(batch):
    raw_images = [item[0] for item in batch]
    images = torch.stack([item[1] for item in batch])
    target = [item[2] for item in batch]
    return [raw_images, images, target]