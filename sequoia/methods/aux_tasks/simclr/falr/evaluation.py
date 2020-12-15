from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import Config, ExperimentType, HParams
from .models import MoCo
from .utils import log


@torch.no_grad()
def prepare_xy(encoder, loader):
  encoder.eval()

  embeddings = []
  for data, target in loader:
    data = data.cuda()
    h = encoder(data)
    embeddings.append((h.cpu().numpy(), target.numpy()))

  X = np.concatenate([x[0] for x in embeddings])
  y = np.concatenate([x[1] for x in embeddings])
  return X, y


def evaluate_logistic(X, y, Xt, yt, epoch: int, log_wandb: bool) -> None:
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  Xt = scaler.transform(Xt)

  clf = LogisticRegression(
    random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000, verbose=0,
  ).fit(X, y)
  
  log({
    'eval/linear/train': np.mean(clf.predict(X) == y),
    'eval/linear/test': np.mean(clf.predict(Xt) == yt),
  }, epoch, 'eval', log_wandb)


def background_logistic(hp:HParams, cfg: Config, q: mp.Queue):
  if cfg.log_wandb:
    wandb.init(project="falr", config=hp.as_dict, group=hp.md5, job_type='background')
  item = q.get()
  while item is not None:
    evaluate_logistic(*item)
    item = q.get()


@torch.no_grad()
def evaluate_classifier(hp: HParams, encoder, projector, classifier, loader, epoch: int, log_wandb: bool, dataset_name: str):
  encoder.eval(); projector.eval(); classifier.eval()

  loss = correct = 0

  for data, target in loader:
    data, target = data.cuda(), target.cuda()
    if hp.experiment == ExperimentType.SUCCESSIVE:
      prediction = classifier(projector(encoder(data)))
    else:
      prediction = classifier(encoder(data))
    loss += F.cross_entropy(prediction, target, reduction="sum")
    prediction = prediction.max(1)[1]
    correct += prediction.eq(target.view_as(prediction)).sum().item()

  loss /= len(loader.dataset)

  percentage_correct = correct / len(loader.dataset)

  log({
    f'eval/ce_loss/{dataset_name}': loss,
    f'eval/acc/{dataset_name}': percentage_correct,
  }, epoch, 'eval', log_wandb)


def evaluate_features(hp: HParams, encoder, projector, classifier, train_loader, test_loader, epoch: int, log_wandb: bool, background_queue: Optional[mp.Queue] = None) -> None:
  encoder.eval(); projector.eval(); classifier.eval()
  if hp.experiment not in {ExperimentType.CONTRASTIVE, ExperimentType.CLASS_CONTRASTIVE}:
    evaluate_classifier(hp, encoder, projector, classifier, train_loader, epoch, log_wandb, 'train')
    evaluate_classifier(hp, encoder, projector, classifier, test_loader, epoch, log_wandb, 'test')
  X, y = prepare_xy(encoder, train_loader)
  Xt, yt = prepare_xy(encoder, test_loader)
  if background_queue is None:
    evaluate_logistic(X, y, Xt, yt, epoch, log_wandb)
  else:
    background_queue.put((X, y, Xt, yt, epoch, log_wandb))

def evaluate_moco_features(hp: HParams, moco: MoCo, train_loader, test_loader, epoch: int, log_wandb: bool, background_queue: Optional[mp.Queue] = None) -> None:
  moco.eval()
  X, y = prepare_xy(moco.encoder_q, train_loader)
  Xt, yt = prepare_xy(moco.encoder_q, test_loader)
  if background_queue is None:
    evaluate_logistic(X, y, Xt, yt, epoch, log_wandb)
  else:
    background_queue.put((X, y, Xt, yt, epoch, log_wandb))
