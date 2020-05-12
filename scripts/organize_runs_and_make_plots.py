from pathlib import Path
from collections import defaultdict
import shutil
import argparse
from typing import Set, List, Tuple, Dict
from distutils.dir_util import copy_tree

  
import os, shutil


def copy_over(source_dir: Path, results_dir: Path, source_runs: str, groups: List[str]) -> Tuple[Dict[Path, List[Path]], Dict[Path, List[Path]]]:
    mh_result_dir = results_dir / source_runs / "multihead"
    mh_d_result_dir = results_dir / source_runs / "multihead_detached"
    
    # # Create the directories if they don't exist
    mh_result_dir.mkdir(parents=True, exist_ok=True)
    mh_d_result_dir.mkdir(parents=True, exist_ok=True)
    
    mh_d_runs: Set[Path] = set(source_dir.glob(f"{source_runs}_mh_d_*"))
    mh_runs: Set[Path] = set(source_dir.glob(f"{source_runs}_mh_*")) - mh_d_runs
    
    # runs_per_group: Dict[str, List[Path]] = defaultdict(list)
    # print(groups)
    # for run in mh_d_runs:
    #     if run.
    
    # print(*mh_runs, sep="\n\t")
    # print("Multihead Detached runs:")
    # print(*mh_d_runs, sep="\n\t")
    mh_paths: Dict[Path, List[Path]] = defaultdict(list)
    mh_d_paths: Dict[Path, List[Path]] = defaultdict(list)

    def get_new_path(result_dir: Path, run: Path, detached: bool=False) -> Path:
        shorter_name = run.name.replace(f"{source_runs}_mh_{'d_' if detached else ''}", "")
        group_dir = "default"
        # Group runs together if they contain one of the strings in 'groups'
        for group_name in groups:
            if group_name in shorter_name:
                shorter_name = shorter_name.replace(group_name, "")
                group_dir = group_name.rstrip("_")
                break
        new_path = result_dir / group_dir / shorter_name
        return new_path

    print("Multihead runs:")
    for run in mh_runs:
        new_path = get_new_path(mh_result_dir, run, detached=False)
        group = new_path.parent

        if new_path.exists():
            print("\t", group, f"(Skipping run {run.name})")
        else:
            print("\t", group, run, "-->", new_path)
            shutil.copytree(run, new_path, symlinks=False)
        mh_paths[group].append(new_path)
    
    print("Multihead (detached) runs:")
    for run in mh_d_runs:
        new_path = get_new_path(mh_d_result_dir, run, detached=True)
        group = new_path.parent

        if new_path.exists():
            print("\t", group, f"Skipping run {run}")
        else:
            print("\t", group, run, "-->", new_path)
            shutil.copytree(run, new_path, symlinks=False)
        mh_d_paths[group].append(new_path)

    return mh_paths, mh_d_paths

from simple_parsing import ArgumentParser, list_field
from dataclasses import dataclass

@dataclass
class Options:
    """ Options for organizing and creating all plots. """
    # Directory which contains all the runs.
    all_runs_dir: Path = Path("E:/Google Drive/SSCL/results/beluga")
    # Directory where all the runs should be organized into.
    organized_dir: Path = Path("E:/Google Drive/SSCL/results/beluga_organized")
    # Directory where figures should be auto-created.
    figures_dir: Path = Path("E:/Google Drive/SSCL/figures/auto")

    # The names of run settings (i.e, prefixes of the runs) to consider.
    settings: List[str] = list_field("cifar100-20c", "cifar100-10c", "cifar10", "mnist", "fashion-mnist")
    
    # Additional Subgroups to cluster runs into. (Runs whose name doesn't
    # contain any of the following will have a group name of 'default'.)
    subgroups: List[str] = list_field("pretrained_ue100_se10_", "pretrained_", "ue20_se10_")
   
    def copy_and_plot(self, experiment: str, run_names: str):
        from make_oml_plot import OmlFigureOptions
        mh_paths, mh_d_paths = copy_over(self.all_runs_dir / experiment, self.organized_dir / experiment, run_names, self.subgroups)
        for group, runs in mh_paths.items():
            path = Path(run_names) / group.name / f"{run_names}_{group.name}_multihead.png"
            OmlFigureOptions(
                runs=[str(group / "*")],
                out_path=self.figures_dir / path,
                exit_after=False,
                add_ntasks_prefix=group.name != "default",
                title=str(path),
            )
        for group, runs in mh_d_paths.items():
            path = Path(run_names) / group.name / f"{run_names}_{group.name}_multihead_detached.png"
            OmlFigureOptions(
                runs=[str(group / "*")],
                out_path=self.figures_dir / path,
                exit_after=False,
                add_ntasks_prefix=group.name != "default",
                title=str(path),
            )

    def __call__(self):
        # import subprocess
        # from shlex import split
        # out_dir = self.all_runs_dir
        # args = split(f"rsync -r --update --verbose normandf@beluga.computecanada.ca:/scratch/normandf/SSCL/ '{out_dir}'")
        # print(args)
        # proc = subprocess.run(args)
        # res = proc.communicate()
        # exit()
        experiment = "TaskIncremental"
        for setting in self.settings:
            self.copy_and_plot(experiment, setting)


if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    options = parser.parse_args().options

    options()
