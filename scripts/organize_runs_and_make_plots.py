from pathlib import Path
from collections import defaultdict
import shutil
import argparse
from typing import Set, List, Tuple, Dict
from distutils.dir_util import copy_tree
import shlex
import os, shutil
import subprocess

from simple_parsing import ArgumentParser, list_field, choice, field
from dataclasses import dataclass
from .make_oml_plot import OmlFigureOptions


DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "E:/Google Drive/"))
print(f"Data dir: {DATA_DIR}")


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



@dataclass
class ScpOptions:
    name: str = "default"
    # input dir
    input_dir: str = "/scratch/$USER/SSCL/"
    # Which server address to use.
    address: str = "beluga.computecanada.ca"
    # which port
    port: int = 22

    exclude: List[str] = list_field("samples", "wandb")

    def run(self, output_dir: Path=DATA_DIR/"results", user: str="normandf"):
        input_dir = self.input_dir.replace("$USER", user)
        if not input_dir.endswith("/"):
            input_dir += "/"
        if "server.mila.quebec" in self.address:
            args = shlex.split(
                f"scp -r -P 2222 {user}@login-1.login.server.mila.quebec:SSCL {output_dir}/"
            )
        else:
            args = shlex.split(
                f"rsync -r --archive --update --verbose " +
                " " + (" ".join(f"--exclude '{e}'" for e in self.exclude)) + " " +
                f"-P {self.port} "
                f"{user}@{self.address}:{input_dir} "
                f":'{output_dir}' "
            )
        print("args: ", args)
        proc = subprocess.run(args)
import contextlib
from io import StringIO

def run(args: Dict) -> OmlFigureOptions:
    s = StringIO()
    with contextlib.redirect_stdout(s):
        obj = OmlFigureOptions(**args)
    s.seek(0)
    # print(s.read())
    return obj


@dataclass
class Options:
    """ Options for organizing and creating all plots. """
    # Wether to skip downloading the runs and just create the plots.
    skip_download: bool = False

    server: ScpOptions = choice({
        "beluga": ScpOptions(
            name="beluga",
            address="beluga.computecanada.ca",
        ),
        "mila": ScpOptions(
            name="mila",
            address="login-1.login.server.mila.quebec",
            port=2222,
            input_dir="/network/home/$USER/SSCL/",
        )
    }, default="mila")
    user: str = field(default=os.environ.get("USER"))

    results_dir: Path = Path(f"{DATA_DIR}/SSCL/results/")

    # Directory where figures should be auto-created.
    figures_dir: Path = Path(f"{DATA_DIR}/SSCL/figures/auto")
    # The names of run settings (i.e, prefixes of the runs) to consider.
    settings: List[str] = list_field("cifar100-20c", "cifar100-10c", "cifar10", "mnist", "fashion-mnist")
    
    # Additional Subgroups to cluster runs into. (Runs whose name doesn't
    # contain any of the following will have a group name of 'default'.)
    subgroups: List[str] = list_field("pretrained_ue100_se10_", "pretrained_", "ue20_se10_", "ue100_se10")
   
    experiments: List[str] = list_field("TaskIncremental")

    def __call__(self):
        self.all_runs_dir = self.results_dir / self.server.name
        if not self.skip_download:
            self.server.run(output_dir=self.all_runs_dir, user=self.user)
        
        if self.server.name == "mila":
            # TODO: SCP command above creates this "SSCL" subfolder inside $DATA_DIR/SSCL/results/mila/
            self.all_runs_dir = self.all_runs_dir / "SSCL"
        
        print(self.all_runs_dir)
        args: List[Dict] = []
        for group_name in self.all_runs_dir.iterdir():
            if group_name.name == "wandb":
                continue
            print(f"Group name: {group_name}")
            args.append(dict(
                runs=[str(group_name / "*")],
                out_path=self.figures_dir / f"{group_name.name}.pdf",
                exit_after=False,
                add_ntasks_prefix=False,
                title=group_name.name,
                show=False,
                maximize_figure=False,
                fig_size_inches=(12, 6),
            ))
        
        import tqdm
        import multiprocessing as mp
        print(f"Creating figures using {mp.cpu_count()} processes.")
        with mp.Pool() as pool:
            for result in pool.imap_unordered(run, args):
                if result.result_figure is not None:
                    print(f"Figure created at path {result.out_path}")
                else:
                    print(f"Couldn't create figure for path {result.out_path}")
        # self.organized_dir = self.results_dir / (self.server.name + "_organised")
    
    

    def copy_and_plot(self, experiment: str, run_names: str):
        from make_oml_plot import OmlFigureOptions
        print("source dir: ", self.all_runs_dir / experiment)
        print("dest dir: ", self.organized_dir / experiment)
        mh_paths, mh_d_paths = copy_over(
            self.all_runs_dir / experiment,
            self.organized_dir / experiment,
            run_names,
            self.subgroups
        )
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

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    options = parser.parse_args().options

    options()
