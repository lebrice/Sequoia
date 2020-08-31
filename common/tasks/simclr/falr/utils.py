import torch
import wandb

from .config import HParams, Config


def save_state(hp: HParams, cfg: Config, epoch: int, encoder, projector, classifier, optimizer, scheduler, loss):
  torch.save({
    'hparams': hp,
    'epoch': epoch,
    'encoder_state_dict': encoder.state_dict(),
    'projector_state_dict': projector.state_dict(),
    'classifier_state_dict': classifier.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss_state_dict': loss.state_dict(),
  }, './checkpoints/' + hp.md5 + '.pkl')
  if cfg.log_wandb:
    wandb.save('./checkpoints/' + hp.md5 + '.pkl')


def load_state(hp: HParams):
  try:
    return torch.load('./checkpoints/' + hp.md5 + '.pkl')
  except FileNotFoundError:
    return None


def log(results: dict, epoch: int, tag: str, log_wandb: bool = False):
  if log_wandb:
    wandb.log(results, step=epoch)

  string = f"[Epoch {epoch}][{tag}]"
  for k in results:
    string += f"[{k} {results[k]}]"
  print(string)
