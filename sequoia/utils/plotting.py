from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


def autolabel(axis, rects: List[plt.Rectangle], bar_height_scale: float=1.):
    """Attach a text label above each bar in *rects*, displaying its height.
    
    Taken from https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        bottom = rect.get_y()
        value = height / bar_height_scale
        if value != 0.0:    
            axis.annotate(
                f"{value:.0%}",
                xy=(rect.get_x() + rect.get_width() / 2, bottom + height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

def maximize_figure():
    fig_manager = plt.get_current_fig_manager()
    try:
        fig_manager.window.showMaximized()
    except:
        try:
            fig_manager.window.state('zoomed') #works fine on Windows!
        except:
            try:
                fig_manager.frame.Maximize(True)
            except:
                print("Couldn't maximize the figure.")


@dataclass
class PlotSectionLabel:
    """ Used to label a section of a plot between `start_step` and `stop_step` with a label of `description`."""
    start_step: int
    stop_step: int
    description: str = ""

    @property
    def middle(self) -> float:
        return (self.start_step + self.stop_step) / 2

    @property
    def width(self) -> int:
        return (self.stop_step - self.start_step)

    def annotate(self, ax: plt.Axes, height: float=-0.1):
        """ Annotate the corresponding region of the axis.
        
        Adds vertical lines at the `start_step` and `end_step` along with a text
        label for the description in between.
            

        Args:
            ax (plt.Axes): An Axis to annotate.
            height (float): The height at which to place the text.
        """
        ax.axvline(self.start_step, linestyle=":", color="gray")
        ax.axvline(self.stop_step,  linestyle=":", color="gray")
        ax.text(self.middle, height, self.description, ha="center")
