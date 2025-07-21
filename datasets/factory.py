from argparse import Namespace
from pathlib import Path
from typing import List, Tuple

import torch.utils.data.datapipes.dataframe.dataframes
from tvmi import transforms
from tvmi.dataloader import ImageFolderLoader


def create_transforms(args: Namespace, mean: List[float], std: List[float]) -> Tuple[
    transforms.Compose, transforms.Compose]:
    """
    Set up the training and validation transforms.

    Args:
        args:
        mean: Per-channel mean for standardizing the data
        std: Per-channel standard deviation for standardizing the data

    Returns:
        Training and validation transforms
    """
    train_transform = transforms.Compose(transforms=[
        transforms.ColorJitter(brightness=args.color_jitter_brightness, contrast=args.color_jitter_contrast,
                               saturation=args.color_jitter_saturation, hue=args.color_jitter_hue),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    val_transform = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    return train_transform, val_transform


def create_dataloader(data_dir: Path, transform: transforms.Compose, ds_ext: str, num_workers: int,
                      shuffle: bool) -> torch.utils.data.DataLoader:
    """
    Create the dataloader for the specified data.

    Args:
        data_dir: Data location
        transform: Transforms to use on the mini-batches
        ds_ext: Image extension
        num_workers: Number of workers for multiprocessing
        shuffle: Whether to shuffle the data

    Returns:
        Dataloader over the specified data
    """
    ds = ImageFolderLoader(root=data_dir, transform=transform, ds_ext=ds_ext)

    return torch.utils.data.DataLoader(dataset=ds, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)
