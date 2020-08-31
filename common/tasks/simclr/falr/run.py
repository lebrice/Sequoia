import os
from pathlib import Path

import papermill as pm
import torch.multiprocessing as mp
import wandb

from .config import Config, HParams
from .experiment import experiment
from .experiment_moco import experiment as moco_experiment

# Make sure to run the data preparation script first
# ./prepare.sh

if __name__=='__main__':
  mp.set_start_method('spawn')
  hp = HParams()
  cfg = Config(Path(os.environ['SLURM_TMPDIR'] + '/data'))
  if hp.use_moco:
    moco_experiment(hp, cfg)
  else:
    experiment(hp, cfg)

  if not hp.use_moco and cfg.save_analysis:
    pm.execute_notebook('./analysis/template.ipynb', f'./analysis/{hp.md5}.ipynb', parameters=dict(md5=hp.md5))
    if cfg.log_wandb:
      wandb.save(f'./analysis/{hp.md5}.ipynb')
