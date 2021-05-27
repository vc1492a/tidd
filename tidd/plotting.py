# imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
import seaborn as sns


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


def plot_distribution(tp_lengths: list, fp_lengths: list) -> sns.FacetGrid:
    """
    creates the plot of visualizing the distribution of true and false positives
    :input: list of true positive sequences
    :input: list of false positive sequences
    :input save_path: place where to save the image of the plot

    # TODO: update docstring

    :return plot:

    """
    # first organize the data into a dataframe
    rows = list()

    for i in tp_lengths:
        rows.append([i, "true_positive"])
    for j in fp_lengths:
        rows.append([j, "false_positive"])

    df_sequence_lengths = pd.DataFrame(
        rows,
        columns=["sequence_length", "sequence_type"]
    )

    colors = ["#4A91C2", "#16068A"]
    # Set your custom color palette
    ax = sns.displot(df_sequence_lengths, x="sequence_length", hue="sequence_type", kde=True, multiple="stack",
                        palette=sns.color_palette(colors))

    return ax


# TODO: move to plotting class / file
def plot_classification(event: pd.DataFrame,
                            pass_id: str,
                            sat_annotation: dict,
                            classification_bool: list,
                            classification_confidence: list,
                            save_path: Union[str, Path] = "./output"
                            ) -> plt.figure:
    """
    Creates 3 plots
    Plot of the time series data with markings of the start and end of the anomalous sequence
    Plot of the predictions for each sequence 1 hour
    Plot of the confidence levels for each of the predictions

    # TODO: update docstring

    :input event: DataFrame containing the float data
    :input pass_id: string containing the satellite and ground station
    :input sat_annotation: dictionary containing the start and end of the anomalous sequence
    :input classification_bool: classification of each time step
    :input classification_confidence: confidence of the classification at each time step
    :input save_path: path where the plot images need to be saved
    """
    # combined "detection evaluation" plot
    fig, axs = plt.subplots(3, sharex=False, sharey=False, figsize=(12, 4))
    fig.tight_layout()
    fig.suptitle('Day of Earthquake Predictions by Second of Day (SoD) for ' + pass_id + "\n")

    gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])

    axs[0] = plt.subplot(gs[0])

    # TODO: NOTE the adjustment below for the window size, make dynamic
    sns.lineplot(data=event.iloc[59:, :], x="sod", y=pass_id, ax=axs[0])
    axs[0].set(yticklabels=[])
    axs[0].axvline(x=sat_annotation["start"], linestyle="dotted")  # start
    axs[0].axvline(x=sat_annotation["finish"], linestyle="dotted")  # approx end - 30 minutes later
    # TODo: do we need to revisit data generation / labeling?
    axs[0].margins(x=0)
    axs[0].set_xlabel("")

    axs[1] = plt.subplot(gs[1])
    sns.heatmap(np.array([classification_bool]), cbar=False, cmap="plasma", xticklabels=False, yticklabels=False,
                ax=axs[1])
    axs[1].set_xlabel("Prediction (yellow is anomaly)")

    axs[2] = plt.subplot(gs[2])
    sns.heatmap(np.array([classification_confidence]), cbar=False, cmap="plasma_r", xticklabels=False,
                yticklabels=False, ax=axs[2])  # .replace(0, np.nan)
    axs[2].set_xlabel("Prediction Confidence (yellow is worse)")

    plt.subplots_adjust(left=0.125,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.5)

    return fig

    