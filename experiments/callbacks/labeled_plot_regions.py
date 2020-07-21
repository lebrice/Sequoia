"""TODO: Migrate to Pytorch-Lightning. """
# from dataclasses import dataclass
# from .addon import ExperimentAddon
# from contextlib import contextmanager
# from utils.plotting import PlotSectionLabel
# from typing import List
# from simple_parsing import list_field, mutable_field


# @dataclass  # type: ignore
# class LabeledPlotRegionsAddon(ExperimentAddon):
    
#     @dataclass
#     class State(ExperimentAddon.State):
#         """ State object of Experiment, but with added plot regions. """
#         # TODO: Use a list of these objects to add annotated regions in the plot
#         # enclosed by vertical lines with some text, for instance "task 0", etc.
#         plot_sections: List[PlotSectionLabel] = list_field()

#     state: State = mutable_field(State, init=False)

#     @contextmanager
#     def plot_region_name(self, description: str):
#         start_step = self.state.global_step
#         plot_section_label = PlotSectionLabel(
#             start_step=start_step,
#             stop_step=0, # only made temporarily.
#             description=description,
#         )
#         self.state.plot_sections.append(plot_section_label)
#         yield
#         plot_section_label.stop_step = self.state.global_step
        
