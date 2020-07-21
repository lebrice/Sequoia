from pathlib import Path
from typing import Callable

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder, ImageNet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


def eval_dataset_speed(dataset: Dataset) -> None:
    """ Load 10 batches of a dataset, showing speed per iteration with tqdm. """        
    batch_size = 16
    num_workers = 16

    dl = DataLoader(
        dataset,
        batch_size=16,
        num_workers=num_workers,
        pin_memory=True,
    )
    n = 10
    # Consider only the first n batches
    from itertools import islice
    dl_part = islice(dl, n)
    dl_pbar = tqdm(dl_part, total=n, description=f"Dataset Type {type(dataset)}")

    for step, batch in enumerate(dl_pbar):
        pass


if __name__ == "__main__":
    from time import time

    image_transform: Callable = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True),
    ])
    image_folder_root = Path("/network/data1/ImageNet2012_jpeg")
    start = time()
    dataset = ImageFolder(image_folder_root, transform=image_transform)
    print(f"Setup time (ImageFolder): {time() - start:.3}s")

    eval_dataset_speed(dataset)

    torchvision_root = Path("/network/datasets/imagenet.var/imagenet_torchvision")
    start = time()
    dataset = ImageNet(torchvision_root, transform=image_transform)
    print(f"Setup time (ImageFolder): {time() - start:.3}s")

    eval_dataset_speed(dataset)
