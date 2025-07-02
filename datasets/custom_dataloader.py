# Source: https://discuss.pytorch.org/t/make-imagefolder-output-the-same-image-twice-with-different-transforms/21887/3
# Modified code slightly from source.

from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None


def make_dataset(root: Path, image_ext: str) -> List[Tuple[str, None]]:
    # TODO Docs
    """

    Args:
        root:
        image_ext:

    Returns:

    """
    images = list(root.rglob(f"*{image_ext}"))
    images_return = []
    for i in range(len(images)):
        images_return.append((str(images[i]), None))
    return images_return


def image_loader(path: Path) -> Image:
    # TODO
    """

    Args:
        path:

    Returns:

    """
    return Image.open(path).convert("RGB")


class ImageFolderLoader(data.Dataset):
    # TODO docs
    def __init__(self, root: Path, transform: transforms.Compose = None, loader: Callable = image_loader,
                 ds_ext: str = "5x", hr: str = "10x", image_ext: str = ".png") -> None:
        """

        Args:
            root:
            transform:
            loader:
            ds_ext:
            hr:
        """
        self.root = root
        self.imgs = make_dataset(root=root, image_ext=image_ext)
        self.transform = transform
        self.loader = loader
        self.ds_ext = ds_ext
        self.hr = hr

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.imgs[index][0]
        img_lr = self.loader(self.root.joinpath(path))
        img_hr = self.loader(str(self.root.joinpath(path)).replace(self.ds_ext, self.hr))

        return self.transform(img_lr, img_hr)

    def __len__(self):
        return len(self.imgs)


