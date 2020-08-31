from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Tuple


class ExperimentType(Enum):
  CLASSIFICATION = 1
  AUGMENTED_CLASSIFICATION = 2
  CONTRASTIVE = 3
  SUCCESSIVE = 4
  BRANCH = 5
  CLASS_CONTRASTIVE = 6


@dataclass
class HParams:
  seed: int = 2022
  cifar: int = 10
  image_size: int = 32
  colour_distortion: float = 0.5
  batch_size: int = 1024 # 256, 512, 1024, 2048, 4096 evaluated in paper
  xent_temp: float = 0.5 # 0.1, 0.5, 1.0 evaluated in paper
  weight_decay: float = 1e-6
  max_lr: float = 0.25 # 0.5, 1.0, 1.5 evaluated in paper
  warmup_epochs: int = 1
  cooldown_epochs: int = 190 # 90, 190, 290, 390, 490, 590, 690, 790, 890, 990 evaluated in paper
  use_lr_decay: bool = False
  resnet_depth: int = 2
  resnet_width: int = 1
  resnet_stacks: Tuple = (64,128,256,512)
  proj_dim: int = 128
  linear_projection: bool = False
  double_augmentation: bool = False
  use_bilinear_loss: bool = False
  use_moco: bool = True
  # classification, augmented_classification, contrastive, successive, branch
  experiment: ExperimentType = ExperimentType.CONTRASTIVE

  def __post_init__(self):
    assert 0 < self.xent_temp <= 1.0
    assert self.cifar in (10, 100)
    self._repr_dim: int = self.resnet_stacks[-1]
    
  @property
  def repr_dim(self) -> int:
    return self._repr_dim
  
  @repr_dim.setter
  def repr_dim(self, value: int) -> None:
    self._repr_dim = value

  @property
  def md5(self):
    return hashlib.md5(str(self).encode('utf-8')).hexdigest()
  
  @property
  def as_dict(self):
    d = asdict(self)
    d['experiment'] = d['experiment'].name
    return d


@dataclass(eq=True, frozen=True)
class Config():
  data_path: Path
  num_workers: int = 0
  pin_memory: bool = False
  log_wandb: bool = True
  evaluation_epoch_freq: int = 20
  checkpoint_epoch_freq: int = 10
  evaluate_background: bool = True
  save_checkpoints: bool = True
  save_analysis: bool = True
