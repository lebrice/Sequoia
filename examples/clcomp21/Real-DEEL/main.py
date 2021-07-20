from sequoia.common.config import WandbConfig
import os
from dataclasses import make_dataclass, fields
from pathlib import Path


if __name__ == "__main__":
    from sequoia.common import Config
    from sequoia.settings import ClassIncrementalSetting
    from simple_parsing import ArgumentParser

    from Methods import METHODS_MAPPING
    from sequoia.common.hparams import HyperParameters

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
        setting = ClassIncrementalSetting(
            dataset="mnist",
            nb_tasks=5,
            monitor_training_performance=True,
            wandb=wandb_config,
            batch_size=16,
        )
    else:
        #     - "HARD": Class-Incremental Synbols, more challenging.
        #     NOTE: This Setting is very similar to the one used for the SL track of the
        #     competition.
        # from sequoia.client.setting_proxy import SettingProxy
        # setting = SettingProxy(ClassIncrementalSetting, "sl_track.yaml")
        setting = ClassIncrementalSetting(
            dataset="synbols",
            nb_tasks=method.hparams.sl_nb_tasks,
            known_task_boundaries_at_test_time=False,
            monitor_training_performance=True,
            batch_size=method.hparams.batch_size,
            num_workers=4,
            wandb=wandb_config,
        )
    #NOTE: can also use pass a `Config` object to `setting.apply`. This object has some
    # configuration options like device, data_dir, etc.
    results = setting.apply(
        method, config=Config(data_dir=os.path.join(args.hparams.output_dir, "data"))
    )
