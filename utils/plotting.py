import matplotlib.pyplot as plt
from typing import List


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
