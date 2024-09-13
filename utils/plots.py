import os
import numpy as np

import matplotlib.pyplot as plt

ALPHAS = ["0.1", "0.3", "0.5", "0.7", "1.0"]

def round_to_nearest(value, base=50):
    return round(value / base) * base

# TODO: Check if this plot makes sense
def plot_capacity_effect(ax, results_dir, save_path=None, label=None):
    """plot the effect of the interpolation parameter on the test accuracy

    :param ax:

    :param results_dir: directory storing the experiment results

    :param save_path: directory to save the plot, default is `None`, if not provided the plot is not saved

    :param label: label of the plot, default is None

    """
    all_scores = np.load(os.path.join(results_dir, "all_scores.npy"))
    n_test_samples = np.load(os.path.join(results_dir, "n_test_samples.npy"))
    weights_grid = np.load(os.path.join(results_dir, "weights_grid.npy"))
    capacities_grid = np.load(os.path.join(results_dir, "capacities_grid.npy"))

    average_scores = np.nan_to_num(all_scores).sum(axis=0) / n_test_samples.sum()

    # TODO: Check original code - possibly wrong
    # for jj, capacity in enumerate(capacities_grid):
    #     accuracies = average_scores[:, jj]
    #     ax.plot(
    #         weights_grid,
    #         accuracies,
    #         linewidth=5.0,
    #         label=label,
    #     )

    for jj, weight in enumerate(weights_grid):
        accuracies = average_scores[jj, :]
        ax.plot(
            capacities_grid,
            accuracies,
            linewidth=5.0,
            label=round(weight, 1),
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ax.grid(True, linewidth=2)

        ax.set_ylabel("Accuracy", fontsize=50)
        ax.set_xlabel("Capacity", fontsize=50)
        ax.tick_params(axis='both', labelsize=25)
        ax.set_xticks(np.arange(0, 1. + 1e-6, 0.1))

        ax.legend(fontsize=25)

        plt.savefig(save_path, bbox_inches='tight')


def plot_weight_effect(results_dir, save_path=None):
    """

    :param results_dir:
    :param save_path:
    :return:

    """
    all_scores = np.load(os.path.join(results_dir, "all_scores.npy"))
    n_train_samples = np.load(os.path.join(results_dir, "n_train_samples.npy"))
    n_test_samples = np.load(os.path.join(results_dir, "n_test_samples.npy"))
    weights_grid = np.load(os.path.join(results_dir, "weights_grid.npy"))
    capacities_grid = np.load(os.path.join(results_dir, "capacities_grid.npy"))

    average_scores = np.nan_to_num(all_scores).sum(axis=0) / n_test_samples.sum()
    # The capacity is normalized w.r.t. the initial size of the client’s (train) dataset partition (like in kNN-per) 
    average_n_train_samples = np.mean(n_train_samples)

    #Debug
    # print("all_scores: ", all_scores.shape)
    # print("n_train_samples: ", n_train_samples.shape, n_train_samples)
    # print("average n_train_samples: ", average_n_train_samples.shape, average_n_train_samples)
    # print("n_test_samples: ", n_test_samples.shape, n_test_samples)
    # print("weights_grid: ", weights_grid.shape, weights_grid)
    # print("average_scores: ", average_scores.shape, average_scores)

    fig, ax = plt.subplots(figsize=(12, 10))

    accuracies = average_scores[:, round(len(capacities_grid)*0.25)]
    ax.plot(
        weights_grid,
        100 * accuracies,
        linewidth=5.0,
        label=fr"$\bar{{n}}_{{m}} \approx {int(round_to_nearest(average_n_train_samples * capacities_grid[round(len(capacities_grid)*0.25)]))}$"
    )

    accuracies = average_scores[:, round(len(capacities_grid)*0.5)] #[:, 10]
    ax.plot(
        weights_grid,
        100 * accuracies,
        linewidth=5.0,
        linestyle="dashdot",
        label=fr"$\bar{{n}}_{{m}} \approx {int(round_to_nearest(average_n_train_samples * capacities_grid[round(len(capacities_grid)*0.5)]))}$"
    )

    accuracies = average_scores[:, round(len(capacities_grid)*0.75)] #[:, 25]
    ax.plot(
        weights_grid,
        100 * accuracies,
        linewidth=5.0,
        linestyle="dashed",
        label=fr"$\bar{{n}}_{{m}} \approx {int(round_to_nearest(average_n_train_samples * capacities_grid[round(len(capacities_grid)*0.75)]))}$"
    )

    accuracies = average_scores[:, -1]
    ax.plot(
        weights_grid,
        100 * accuracies,
        linewidth=5.0,
        linestyle="dotted",
        label=fr"$\bar{{n}}_{{m}} \approx {int(round_to_nearest(average_n_train_samples * capacities_grid[-1]))}$"
    )

    ax.grid(True, linewidth=2)

    ax.set_ylabel("Test accuracy", fontsize=50)
    ax.set_xlabel(r"$\lambda$", fontsize=50)
    ax.tick_params(axis='both', labelsize=25)

    ax.legend(fontsize=25)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_hetero_effect(results_dir, save_path=None):
    _, ax = plt.subplots(figsize=(12, 10))

    for alpha in ALPHAS:

        current_dir = os.path.join(results_dir, f"n_neighbors_7_alpha_{alpha}")
        label = r"$\alpha$={}".format(alpha)

        plot_capacity_effect(ax, results_dir=current_dir, label=label)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ax.grid(True, linewidth=2)

        ax.set_ylabel("Accuracy", fontsize=50)
        ax.set_xlabel("Capacity", fontsize=50)
        ax.tick_params(axis='both', labelsize=25)
        ax.set_xticks(np.arange(0, 1. + 1e-6, 0.1))

        ax.legend(fontsize=25)

        plt.savefig(save_path, bbox_inches='tight')

    else:
        plt.show()

    plt.close()
