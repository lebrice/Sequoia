"""TODO: Re-enable the wandb stuff (disabled for now).
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import *

from pytorch_lightning.loggers import WandbLogger
from simple_parsing import field, list_field

import wandb
from sequoia.utils.logging_utils import get_logger
from sequoia.utils.serialization import Serializable


def patched_monitor():
    vcr = wandb.util.get_module(
        "gym.wrappers.monitoring.video_recorder",
        required="Couldn't import the gym python package, install with pip install gym",
    )
    print(f"Using patched version of `wandb.gym.monitor()`")
    if hasattr(vcr.ImageEncoder, "orig_close"):
        print(f"wandb.gym.monitor() has already been called.")
        return
    else:
        vcr.ImageEncoder.orig_close = vcr.ImageEncoder.close

    def close(self):
        vcr.ImageEncoder.orig_close(self)
        m = re.match(r".+(video\.\d+).+", self.output_path)
        if m:
            key = m.group(1)
        else:
            key = "videos"
        wandb.log({key: wandb.Video(self.output_path)})

    vcr.ImageEncoder.close = close
    wandb.patched["gym"].append(["gym.wrappers.monitoring.video_recorder.ImageEncoder", "close"])


import wandb.integration.gym

wandb.integration.gym.monitor = patched_monitor


# GYM_MONITOR = os.environ.get("GYM_MONITOR", "")
# if not GYM_MONITOR:
#     wandb.gym.monitor()
#     os.environ["GYM_MONITOR"] = "True"
# else:
#     assert False, "importing this a second time?"

logger = get_logger(__file__)


@dataclass
class WandbConfig(Serializable):
    """Set of configurations options for calling wandb.init directly."""

    # Which user to use
    entity: str = ""

    # project name to use in wandb.
    project: str = ""

    # Name used to easily group runs together.
    # Used to create a parent folder that will contain the `run_name` directory.
    # A unique string shared by all runs in a given group
    # Used to create a parent folder that will contain the `run_name` directory.
    group: Optional[str] = None
    # Wandb run name. If None, will use wandb's automatic name generation
    run_name: Optional[str] = None

    # Identifier unique to each individual wandb run. When given, will try to
    # resume the corresponding run, generates a new ID each time.
    run_id: Optional[str] = None

    # An run number is used to differentiate different iterations of the same experiment.
    # Runs with the same name can be later grouped with wandb to produce stderr plots.
    # TODO: Could maybe use the run_id instead?
    run_number: Optional[int] = None

    # Path where the wandb files should be stored. If the 'WANDB_DIR'
    # environment variable is set, uses that value. Otherwise, defaults to
    # the value of "<log_dir_root>/wandb"
    wandb_path: Optional[Path] = (
        Path(os.environ["WANDB_DIR"]) if "WANDB_DIR" in os.environ else None
    )

    # Tags to add to this run with wandb.
    tags: List[str] = list_field()

    # Notes about this particular experiment. (will be logged to wandb if used.)
    notes: Optional[str] = None

    # Root Logging directory.
    log_dir_root: Path = Path("results")

    monitor_gym: bool = True

    # Wandb api key. Useful for preventing the login prompt from wandb from appearing
    # when running on clusters or docker-based setups where the environment variables
    # aren't always shared.
    wandb_api_key: Optional[Union[str, Path]] = field(
        default=os.environ.get("WANDB_API_KEY"),
        to_dict=False,  # Do not serialize this field.
        repr=False,  # Do not show this field in repr().
    )

    # Run offline (data can be streamed later to wandb servers).
    offline: bool = False
    # Enables or explicitly disables anonymous logging.
    anonymous: bool = False
    # Sets the version, mainly used to resume a previous run.
    version: Optional[str] = None

    # Save checkpoints in wandb dir to upload on W&B servers.
    log_model: bool = False

    # Class variables used to check wether wandb.login has already been called or not.
    logged_in: ClassVar[bool] = False
    key_configured: ClassVar[bool] = False

    @property
    def log_dir(self):
        return self.log_dir_root.joinpath(
            (self.project or ""),
            (self.group or ""),
            (self.run_name or "default"),
            (f"run_{self.run_number}" if self.run_number is not None else ""),
        )

    def wandb_login(self) -> bool:
        """Calls `wandb.login()`.

        Returns
        -------
        bool
            If the key is configured.
        """
        key = None
        if self.wandb_api_key is not None and self.project:
            if Path(self.wandb_api_key).is_file():
                key = Path(self.wandb_api_key).read_text()
            else:
                key = str(self.wandb_api_key)
            assert isinstance(key, str)

        cls = type(self)
        if not cls.logged_in:
            cls.key_configured = wandb.login(key=key)
            cls.logged_in = True
        return cls.key_configured

    def wandb_init_kwargs(self) -> Dict:
        """Return the kwargs to pass to wandb.init()"""
        if self.run_name is None:
            # TODO: Create a run name using the coefficients of the tasks, etc?
            # At the moment, if no run name is given, the 'random' name from wandb is used.
            pass
        if self.wandb_path is None:
            self.wandb_path = self.log_dir_root / "wandb"
        self.wandb_path.mkdir(parents=True, mode=0o777, exist_ok=True)
        return dict(
            dir=str(self.wandb_path),
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            id=self.run_id,
            group=self.group,
            notes=self.notes,
            reinit=True,
            tags=self.tags,
            resume="allow",
            monitor_gym=self.monitor_gym,
        )

    def wandb_init(self, config_dict: Dict = None) -> wandb.wandb_run.Run:
        """Executes the call to `wandb.init()`.

        TODO(@lebrice): Not sure if it still makes sense to call `wandb.init`
        ourselves when using Pytorch Lightning, should probably ask @jeromepl
        for advice on this.

        Args:
            config_dict (Dict): The configuration dictionary. Usually obtained
            by calling `to_dict()` on a `Serializable` dataclass, or `asdict()`
            on a regular dataclass.

        Returns:
            wandb.wandb_run.Run: Whatever gets returned by `wandb.init()`.
        """

        logger.info(f"Wandb run id: {self.run_id}")
        logger.info(
            f"Using wandb. Group name: {self.group} run name: {self.run_name}, "
            f"log_dir: {self.log_dir}"
        )
        self.wandb_login()

        init_kwargs = self.wandb_init_kwargs()
        init_kwargs["config"] = config_dict

        run = wandb.init(**init_kwargs)
        logger.info(f"Run: {run}")
        if run:
            if self.run_name is None:
                self.run_name = run.name
            # run.save()
            if run.resumed:
                # TODO: add *proper* wandb resuming, probaby by using @nitarshan 's md5 id cool idea.
                # wandb.restore(self.log_dir / "checkpoints")
                pass
        return run

    def make_logger(self, wandb_parent_dir: Path = None) -> WandbLogger:
        logger.info(f"Creating a WandbLogger with using options {self}.")
        self.wandb_login()
        wandb_logger = WandbLogger(
            name=self.run_name,
            save_dir=str(wandb_parent_dir) if wandb_parent_dir else None,
            offline=self.offline,
            id=self.run_id,
            anonymous=self.anonymous,
            version=self.version,
            project=self.project,
            tags=self.tags,
            log_model=self.log_model,
            entity=self.entity,
            group=self.group,
            monitor_gym=self.monitor_gym,
            reinit=True,
        )
        return wandb_logger
