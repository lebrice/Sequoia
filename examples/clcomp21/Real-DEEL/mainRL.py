""" Example Method for the SL track: Uses a simple classifier and DER mechanism.

"""


from sequoia.common.config import WandbConfig
import os
from dataclasses import make_dataclass, fields
from pathlib import Path

from sequoia.common import Config
from simple_parsing import ArgumentParser

from Methods import METHODS_MAPPING

if __name__ == "__main__":
    from sequoia.common import Config
    from sequoia.settings import IncrementalRLSetting
    from sequoia.client import SettingProxy

    from simple_parsing import ArgumentParser

    # This code creates a virtual display to draw game images on.
    # If you are running locally, you can just ignore it
    


    parser = ArgumentParser()
    hparams = {}
    for Method in METHODS_MAPPING.values():
        [
            hparams.update({hparam.name: (hparam.name, hparam.type, hparam)})
            for hparam in fields(Method.HParams())
        ]

    hparams = make_dataclass("dynamic", tuple(hparams.values()))
    parser.add_arguments(hparams, "hparams")

    args, unknown = parser.parse_known_args()
    assert args.hparams.cl_method_name in METHODS_MAPPING


    from pyvirtualdisplay import Display
    def create_display():
        display = Display(visible=0, size=(1400, 900))
        display.start()
        if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY")) == 0:
            # FIXME ! in jupyter notebook means a bash command line
            os.system("/bin/bash -c ../xvfb start")
            # FIXME %env is a magic keyword should be replaced with equivalent python command
            # %env DISPLAY=:1
            os.environ['DISPLAY'] = 1
    create_display()

    print(args)
    Method = METHODS_MAPPING[args.hparams.cl_method_name]
    method = Method.from_argparse_args(args)
    # prepare output path
    if not (os.path.isdir(args.hparams.output_dir)):
        os.makedirs(args.hparams.output_dir)
        os.mkdir(os.path.join(args.hparams.output_dir, "wandb"))
        os.mkdir(os.path.join(args.hparams.output_dir, "data"))

    wandb_config = None
    if args.hparams.wandb or args.hparams.wandb_logging:
        wandb_config = WandbConfig(
            project=args.hparams.wandb_project,
            entity=args.hparams.wandb_entity,
            wandb_api_key=args.hparams.wandb_api,
            run_name=args.hparams.wandb_run_name,
            wandb_path=Path(os.path.join(args.hparams.output_dir, "wandb")),
        )
 
    if args.hparams.debug_mode:
        os.environ["WANDB_MODE"] = "dryrun"
        setting = IncrementalRLSetting(
            dataset="CartPole-v0",
            observe_state_directly=True,
            monitor_training_performance=True,
            nb_tasks=1,
            steps_per_task=1_000,
            test_steps=2000,
            wandb=wandb_config,
        )
    else:
        #     NOTE: This Setting is very similar to the one used for the SL track of the
        #     competition.
        setting = SettingProxy(IncrementalRLSetting, "rl_track.yaml")

    # NOTE: can also use pass a `Config` object to `setting.apply`. This object has some
    # configuration options like device, data_dir, etc.
    results = setting.apply(method, config=Config(data_dir="data"))
