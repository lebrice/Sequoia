import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import *

from simple_parsing import ArgumentParser, list_field

from .make_oml_plot import get_nonempty_run_dirs, is_log_dir, is_run_dir
from .organize_runs_and_make_plots import DATA_DIR


def merge_results(server_dirs: List[Path], out_dir: Path) -> None:
    """Merges the results for from the different servers (mila, beluga, etc)
    
    if given dirs A and B with structure:
    ```
    A
    └── baseline
        ├── run_0
        ├── run_1
        └── run_2
    B
    └── baseline
        ├── run_0
        ├── run_1
        └── run_7
    ```
    then the results will have
    ```
    out_dir
    └── baseline
        ├── run_0
        ├── run_1
        ├── run_2
        ├── run_3
        ├── run_4
        └── run_5
    ```

    Note: doesn't preserve any kind of ordering.   

    Args:
        *server_dirs (Path): Paths to the collections of runs. (parent folder of the run groups, ex. TaskIncremental)
        out_dir (Path): [description]
    """
    from functools import partial
    run_groups: DefaultDict[str, DefaultDict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    # key: Run group
    # key2: Log dir
    # Values: List of paths for each run_dir.

    for source_dir in server_dirs:
        for run_group_path in source_dir.iterdir():
            run_group = run_group_path.name

            for log_dir in run_group_path.iterdir():
                run_name = log_dir.name
                # print(f"Run name: {run_name}")

                runs_to_add = list(get_nonempty_run_dirs(log_dir))
                # print(f"Runs to add: {runs_to_add}")
                if runs_to_add:
                    run_groups[run_group][run_name].extend(runs_to_add)
    
    for run_group, name_to_runs in run_groups.items():
        print(f"Run group: {run_group}")
        for name, run_dirs in name_to_runs.items():
            print(f"\tRun name: {name}")
            # The directory where all the runs will be merged            
            merged_dir = out_dir / run_group / name

            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            merged_dir.mkdir(parents=True, exist_ok=False)

            print(f"\tMerged dir: {merged_dir}")
            for i, run_dir in enumerate(sorted(run_dirs)):
                destination = merged_dir / f"run_{i}"
                print(f"\t{run_dir} --> {destination}")
                shutil.copytree(run_dir, destination)


@dataclass
class MergeResults:
    server_dirs: List[Path] = list_field(DATA_DIR / "SSCL" / "results" / "mila" / "SSCL",
                                         DATA_DIR / "SSCL" / "results" / "beluga")
    out_dir: Path = DATA_DIR / "SSCL" / "results" / "merged"

    def __call__(self):
        merge_results(self.server_dirs, self.out_dir)

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(MergeResults, dest="options")
    args = parser.parse_args()
    options: MergeResults = args.options
    options()

