# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union


def gramian_angular_field(array: Union[pd.Series, np.array, list]) -> plt.figure:
    """
    # TODO:
    """

    figure = plt.figure(figsize=(5, 5), frameon=False)

    ax = plt.Axes(figure, [0., 0., 1., 1.])
    ax.set_axis_off()
    figure.add_axes(ax)

    figure = plt.imshow(array[0], cmap='viridis', origin='lower')

    x_axis = figure.axes.get_xaxis()
    x_axis.set_visible(False)

    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)

    return figure
