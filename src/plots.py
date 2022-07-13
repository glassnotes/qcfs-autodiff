import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "Times"})

# Constants for plotting that are consistent throughout our simulations
R_VALUES = np.arange(0.5, 1.5, 0.1)
LABELS = ["Unmitigated", "1,3", "1,3,5", "1,3,5,7"]
FOLDING_TYPES = ["Unitary", "CNOT"]
COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:pink"]
STYLES = ["o", "x"]
OFFSETS = [-0.015, 0, 0.015, 0.03, 0.045]
ACC_OFFSETS = [0, 0.02]


def make_scatter_plot(ax, true_data, unitary_data, cnot_data, y_axis_label, error_bar_type="std"):
    """Make two-panel scatter plot of the data to compare unitary folding and CNOT folding.

    Args:
        ax (Axes): A pair of matplotlib axes to plot on.
        true_data (array[float]): The exact values of the data being plotted.
        unitary_data (list[array[array[float]]]): Values obtain through unitary
            folding. Each element of the list should be a set of trials for a
            particular set of unitary folding scale factors, separated by value
            of r. For our experients, its shape should be (k, 10, 100) where k
            is the number of different scale factor sets investigated. The first
            element should correspond to unmitigated data.
        cnot_data (list[array[array[float]]]): The same type of data as the previous
            argument, but for CNOT folding.
        error_bar_type (str): Indicates what value to plot for error bars. Either "std"
            or "std-error".
    """
    error_bar_scaling = 1
    if error_bar_type == "std-error":
        error_bar_scaling = np.sqrt(len(unitary_data[0][0]))

    for ax_idx, folding_data in enumerate([unitary_data, cnot_data]):
        ax[ax_idx].scatter(R_VALUES - 0.03, true_data, color="black", label="Exact", marker="v")

        for data_idx, results in enumerate(folding_data):
            ax[ax_idx].errorbar(
                R_VALUES + OFFSETS[data_idx],
                np.mean(results, axis=1),
                yerr=np.std(results, axis=1) / error_bar_scaling,
                label=LABELS[data_idx],
                capsize=4,
                fmt="o",
                color=COLOURS[data_idx],
            )

        ax[ax_idx].tick_params(axis="y", labelsize=14)
        ax[ax_idx].tick_params(top=True, labeltop=False, bottom=True, labelbottom=True)
        ax[ax_idx].tick_params(right=True, direction="in", length=6)
        ax[ax_idx].set_xlabel("$r$", fontsize=16)
        ax[ax_idx].set_title(f"{FOLDING_TYPES[ax_idx]} folding", fontsize=18)
        ax[ax_idx].set_ylabel(y_axis_label, fontsize=16)
        ax[ax_idx].set_xticks(R_VALUES, labels=[f"{x:.1f}" for x in R_VALUES], fontsize=14)
        ax[ax_idx].legend(fontsize=14)


def make_mean_absolute_error_plot(ax, true_data, unitary_data, cnot_data):
    """Plots the mean absolute error of the data, i.e., for each trial,
    we compute the absolute error compare to the exact value, then take the mean.

    Args:
        ax (Axes): A single matplotlib axis to plot on.
        true_data (array[float]): The exact values of the data being plotted.
        unitary_data (list[array[array[float]]]): Values obtain through unitary
            folding. Each element of the list should be a set of trials for a
            particular set of unitary folding scale factors, separated by value
            of r. For our experients, its shape should be (k, 10, 100) where k
            is the number of different scale factor sets investigated. The first
            element should correspond to unmitigated data.
        cnot_data (list[array[array[float]]]): The same type of data as the previous
            argument, but for CNOT folding.
    """
    ax.scatter(
        R_VALUES - ACC_OFFSETS[-1],
        np.mean(np.abs(true_data[:, np.newaxis] - unitary_data[0]), axis=1),
        color=COLOURS[0],
        label="Unmitigated",
        marker="v",
    )

    for type_idx, folding_data in enumerate([unitary_data[1:], cnot_data[1:]]):
        for data_idx, results in enumerate(folding_data):
            ax.scatter(
                R_VALUES + ACC_OFFSETS[type_idx],
                np.mean(np.abs(true_data[:, np.newaxis] - results), axis=1),
                color=COLOURS[1:][data_idx],
                label=FOLDING_TYPES[type_idx] + " " + LABELS[1:][data_idx],
                marker=STYLES[type_idx],
            )

    ax.set_xlabel("$r$", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(top=True, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(right=True, direction="in", length=6)
    ax.set_ylabel("Mean absolute mitigation error", fontsize=16)
    ax.set_xticks(R_VALUES, labels=[f"{x:.1f}" for x in R_VALUES], fontsize=14)
    ax.legend(fontsize=14)


def make_absolute_error_mean_plot(ax, true_data, unitary_data, cnot_data, legend=True):
    """Plots the absolute error of the mean data, i.e., we take the mean over all
    trials, and compute the absolute error.

    Args:
        ax (Axes): A single matplotlib axis to plot on.
        true_data (array[float]): The exact values of the data being plotted.
        unitary_data (list[array[array[float]]]): Values obtain through unitary
            folding. Each element of the list should be a set of trials for a
            particular set of unitary folding scale factors, separated by value
            of r. For our experients, its shape should be (k, 10, 100) where k
            is the number of different scale factor sets investigated. The first
            element should correspond to unmitigated data.
        cnot_data (list[array[array[float]]]): The same type of data as the previous
            argument, but for CNOT folding.
    """
    ax.scatter(
        R_VALUES - ACC_OFFSETS[-1],
        np.abs(true_data - np.mean(unitary_data[0], axis=1)),
        color=COLOURS[0],
        label="Unmitigated",
        marker="v",
    )

    for type_idx, folding_data in enumerate([unitary_data[1:], cnot_data[1:]]):
        for data_idx, results in enumerate(folding_data):
            ax.scatter(
                R_VALUES + ACC_OFFSETS[type_idx],
                np.abs(true_data - np.mean(results, axis=1)),
                color=COLOURS[1:][data_idx],
                label=FOLDING_TYPES[type_idx] + " " + LABELS[1:][data_idx],
                marker=STYLES[type_idx],
            )

    ax.set_xlabel("$r$", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(top=True, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(right=True, direction="in", length=6)
    ax.set_ylabel("Absolute mitigation error of mean", fontsize=16)
    ax.set_xticks(R_VALUES, labels=[f"{x:.1f}" for x in R_VALUES], fontsize=14)
    if legend is True:
        ax.legend(fontsize=14)
