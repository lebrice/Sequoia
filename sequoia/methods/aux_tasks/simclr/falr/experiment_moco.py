import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from .config import Config, ExperimentType, HParams
from .data import get_moco_loaders
from .evaluation import background_logistic, evaluate_moco_features
from .models import MoCo
from .utils import log


def train(moco, loader, optimizer, epoch, cfg: Config) -> None:
  start = time.time()
  
  log({"train/lr": optimizer.param_groups[0]['lr']}, epoch, 'train', cfg.log_wandb)

  moco.train()

  losses = []

  for images, _ in tqdm(loader, leave=False):
    im_q = images[0].cuda(non_blocking=True)
    im_k = images[1].cuda(non_blocking=True)

    output, target = moco(im_q, im_k)
    loss = F.cross_entropy(output, target)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  log({"train/avg_loss": np.mean(losses),}, epoch, 'train', cfg.log_wandb)
  log({'train/runtime': time.time()-start}, epoch, 'train', cfg.log_wandb)


def experiment(hp: HParams, cfg: Config):
  print('Experiment', hp.md5)
  print(torch.get_num_threads(), 'cpu cores available')

  if hp.experiment != ExperimentType.CONTRASTIVE:
    raise RuntimeError("MoCo only supports contrastive experiments for now.")

  torch.manual_seed(1)

  # Dataset
  train_loader, train_eval_loader, test_loader = get_moco_loaders(hp, cfg)

  # Models
  moco = MoCo(hp).cuda()

  # Optimizers and Schedulers
  init_lr = hp.max_lr/(hp.warmup_epochs + 1)
  optimizer = torch.optim.SGD(moco.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-6)
  cosine_scheduler = CosineAnnealingLR(optimizer, hp.cooldown_epochs)

  # Starting Epoch
  epoch = 1

  # Background evaluation setup (has to be before wandb init in this process!)
  background_queue = None
  if cfg.evaluate_background:
    background_queue = mp.Queue()
    background_process = mp.Process(target=background_logistic, args=(hp, cfg, background_queue))
    background_process.start()

  # Wandb  
  if cfg.log_wandb:
    wandb.init(project="falr", config=hp.as_dict, group=hp.md5, job_type='main')

  for epoch in range(epoch, hp.warmup_epochs + hp.cooldown_epochs + 1):
    train(moco, train_loader, optimizer, epoch, cfg)
    
    if epoch <= hp.warmup_epochs:
      optimizer.param_groups[0]['lr'] = min(hp.max_lr, hp.max_lr * (epoch+1)/(hp.warmup_epochs+1)) # Pytorch LambdaLR scheduler is buggy...
    elif hp.use_lr_decay:
      cosine_scheduler.step()
    
    if (epoch == 1) or (epoch % cfg.evaluation_epoch_freq == 0) or (epoch == hp.warmup_epochs + hp.cooldown_epochs):
      evaluate_moco_features(hp, moco, train_eval_loader, test_loader, epoch, cfg.log_wandb, background_queue)

  if cfg.evaluate_background:
    background_queue.put(None)
    background_process.join()
