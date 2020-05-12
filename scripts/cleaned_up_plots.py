from make_oml_plot import OmlFigureOptions
from pathlib import Path
from simple_parsing import ArgumentParser, list_field
from dataclasses import dataclass
from typing import List

def format_label(run_path: Path, current_label: str) -> str:
    just_task_names = (current_label
        # Get rid of the prefix that indicates the number of tasks:
        .replace("0_", "_")
        .replace("1_", "_")
        .replace("2_", "_")
        .replace("3_", "_")
        # Get rid of the coefficients:
        # (usually *_1* or *_01* or *_001* and *_nc_*)
        .replace("1", "_")
        .replace("0", "_")
        .replace("nc", "_")
    )
    return " + ".join(just_task_names.replace("_", " ").split())


@dataclass
class Options:
        
    cleaned_up_dir: Path = Path('E:/Google Drive/SSCL/results/cleaned_up')
    cleaned_up_fig_dir: Path = Path('E:/Google Drive/SSCL/figures/cleaned_up')
    settings: List[str] = list_field("cifar100-20c", "cifar10", "fashion-mnist")
    mhs: List[str] = list_field("multihead", "multihead_detached")

    def __call__(self):
        for setting in self.settings:
            for mh in self.mhs:
                OmlFigureOptions(
                    runs=[str(self.cleaned_up_dir / setting / mh / "*")],
                    out_path=self.cleaned_up_fig_dir / f"{setting}_{mh}.pdf",
                    exit_after=False,
                    add_ntasks_prefix=False,
                    title="",
                    # title=setting + " " + mh.replace("_", " "),
                    maximize_figure=False,
                    fig_size_inches=(12, 5),
                    label_formatting_fn=format_label,
                    legend_position=("upper right" if "cifar100" in setting else "lower left"),
                )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args()
    options: Options = args.options
    options()