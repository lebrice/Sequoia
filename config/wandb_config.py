from utils.logging_utils import get_logger
import os
from dataclasses import dataclass
from pathlib import Path
from typing import *

import wandb

from simple_parsing import field, list_field
from utils.json_utils import Serializable

logger = get_logger(__file__)

@dataclass
class WandbConfig(Serializable):
    # Which user to use
    entity: str = "lebrice"

    project_name: str = "SSCL_replay" # project name to use in wandb.
    # Name used to easily group runs together.
    # Used to create a parent folder that will contain the `run_name` directory. 
    run_group: Optional[str] = None
    # Wandb run name. If None, will use wandb's automatic name generation
    run_name: Optional[str] = None

    # Identifier unique to each individual wandb run. When given, will try to
    # resume the corresponding run, generates a new ID each time. 
    run_id: str = field(default_factory=wandb.util.generate_id)

    # An run number is used to differentiate different iterations of the same experiment.
    # Runs with the same name can be later grouped with wandb to produce stderr plots.
    # TODO: Could maybe use the run_id instead?
    run_number: Optional[int] = None 

    # Path where the wandb files should be stored. If the 'WANDB_DIR'
    # environment variable is set, uses that value. Otherwise, defaults to
    # the value of "<log_dir_root>/wandb"
    wandb_path: Optional[Path] = Path(os.environ['WANDB_DIR']) if "WANDB_DIR" in os.environ else None

    # Tags to add to this run with wandb.
    tags: List[str] = list_field() 

    # Notes about this particular experiment. (will be logged to wandb if used.)
    notes: Optional[str] = None
    
    # Root Logging directory.
    log_dir_root: Path = Path("results")



    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(
            (self.project_name or ""),
            (self.run_group or ""),
            (self.run_name or 'default'),
            (f"run_{self.run_number}" if self.run_number is not None else ""),
        )

    def wandb_init(self, config_dict: Dict) -> wandb.wandb_run.Run:
        if self.run_name is None:
            # TODO: Create a run name using the coefficients of the tasks, etc?
            # At the moment, if no run name is given, the 'random' name from wandb is used.
            pass
        logger.info(f"Using wandb. Experiment name: {self.run_name}")
        
        if self.wandb_path is None:
            self.wandb_path = self.log_dir_root / "wandb"
        self.wandb_path.mkdir(parents=True, mode=0o777, exist_ok=True)

        logger.info(f"Wandb run id: {self.run_id}")
        logger.info(f"Using wandb. Group name: {self.run_group} run name: {self.run_name}, log_dir: {self.log_dir}")

        run = wandb.init( 
            project=self.project_name,
            name=self.run_name,
            id=self.run_id,  
            group=self.run_group,
            config=config_dict,
            dir=str(self.wandb_path),
            notes=self.notes,
            reinit=True,
            tags=self.tags,
            resume="allow",
        )
        logger.info(f"Run: {run}")
        run.save()
        
        if run.resumed:
            # TODO: add *proper* wandb resuming, probaby by using @nitarshan 's md5 id cool idea.
            wandb.restore(self.log_dir / "checkpoints")
            
            pass

        wandb.save(str(self.log_dir / "*"))
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.save_yaml(self.log_dir / "config.yml")

        if self.run_name is None:
            self.run_name = run.name
        
        return run
