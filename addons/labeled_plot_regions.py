from dataclasses import dataclass
from experiment import ExperimentBase
from contextlib import contextmanager
from utils.plotting import PlotSectionLabel


@dataclass  # type: ignore
class LabeledPlotRegionsAddon(ExperimentBase):
    def __post_init__(self):
        super().__post_init__()
        # TODO: Use a list of these objects to add annotated regions in the plot
        # enclosed by vertical lines with some text, for instance "task 0", etc.
        self.plot_sections: List[PlotSectionLabel] = []

    @contextmanager
    def plot_region_name(self, description: str):
        start_step = self.global_step
        yield
        end_step = self.global_step
        plot_section_label = PlotSectionLabel(start_step, end_step, description)
        self.plot_sections.append(plot_section_label)
