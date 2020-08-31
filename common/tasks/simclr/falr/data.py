import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import (
  ColorJitter, Compose, Lambda, RandomApply, RandomGrayscale,
  RandomHorizontalFlip, RandomResizedCrop, ToTensor)

from .config import Config, HParams


class SimCLRAugment(object):
  def __init__(self, hp: HParams, count: int = 2):
    s = hp.colour_distortion
    self.double_augmentation = hp.double_augmentation
    self.count = count
    self.simclr_augment = Compose([
      RandomResizedCrop(hp.image_size),
      RandomHorizontalFlip(),
      RandomApply([
        ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
      ], p=0.8),
      RandomGrayscale(p=0.2),
      ToTensor(),
    ])

  def __call__(self, img: PIL.Image.Image):
    augs = []
    if self.double_augmentation:
      augs.append(self.simclr_augment(img))
    else:
      augs.append(ToTensor()(img))
    for _ in range(self.count-1):
      augs.append(self.simclr_augment(img))

    return torch.stack(augs)


class MoCoAugment(object):
  def __init__(self, hp: HParams):
    s = hp.colour_distortion
    self.simclr_augment = Compose([
      RandomResizedCrop(hp.image_size),
      RandomHorizontalFlip(),
      RandomApply([
        ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
      ], p=0.8),
      RandomGrayscale(p=0.2),
      ToTensor(),
    ])

  def __call__(self, img: PIL.Image.Image):
    return [self.simclr_augment(img), self.simclr_augment(img)]


def tmp2(ts):
  return np.array([ts, ts])


def get_loaders(hp: HParams, cfg: Config):
  train_transform = SimCLRAugment(hp)
  target_transform = Lambda(tmp2)

  if hp.cifar == 10:
    dataset = datasets.CIFAR10
  elif hp.cifar == 100:
    dataset = datasets.CIFAR100
  else:
    raise KeyError
  
  # Crops x Channels x Height x Width
  augmented_train_dataset = dataset(cfg.data_path, train=True, transform=train_transform, target_transform=target_transform, download=False)
  # Channels x Height x Width
  train_dataset = dataset(cfg.data_path, train=True, transform=ToTensor(), download=False)
  # Channels x Height x Width
  test_dataset = dataset(cfg.data_path, train=False, transform=ToTensor(), download=False)

  augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
  train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
  test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
  
  return augmented_train_loader, train_loader, test_loader

def get_moco_loaders(hp: HParams, cfg: Config):
  train_transform = MoCoAugment(hp)

  if hp.cifar == 10:
    dataset = datasets.CIFAR10
  elif hp.cifar == 100:
    dataset = datasets.CIFAR100
  else:
    raise KeyError
  
  # [Channels x Height x Width, Channels x Height x Width]
  augmented_train_dataset = dataset(cfg.data_path, train=True, transform=train_transform, download=False)
  # Channels x Height x Width
  train_dataset = dataset(cfg.data_path, train=True, transform=ToTensor(), download=False)
  # Channels x Height x Width
  test_dataset = dataset(cfg.data_path, train=False, transform=ToTensor(), download=False)

  augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=True)
  train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
  test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
  
  return augmented_train_loader, train_loader, test_loader
