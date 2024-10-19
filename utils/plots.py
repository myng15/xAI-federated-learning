import os
import numpy as np

import matplotlib.pyplot as plt

ALPHAS = ["0.1", "0.3", "0.5", "0.7", "1.0"]

def round_to_nearest(value, base=50):
    return round(value / base) * base

def plot_client_evaluation(results_dir, mode="train", save_path=None):
    """
    Plot client evaluation results either for training or test clients based on the mode.
    
    :param results_dir: Directory where evaluation results are stored
    :param mode: "train" for training clients, "test" for test clients
    :param save_path: Path to save the plot, or None to display the plot
    :return: None
    """
    if mode == "train":
        results_filename = "fedavg_all_client_results.npy"
    elif mode == "test":
        results_filename = "fedavg_all_test_client_results.npy"
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    # Load the evaluation results for the specified mode (train or test clients)
    results_path = os.path.join(results_dir, results_filename)
    eval_rounds_path = os.path.join(results_dir, "fedavg_all_eval_rounds.npy")
    
    if not os.path.exists(results_path):
        print(f"No results found for {mode} clients at {results_path}. Skipping plot.")
        return

    all_client_results = np.load(results_path)
    all_eval_rounds = np.load(eval_rounds_path)

    # Calculate average accuracy per round for train and test datasets
    train_accuracies = []
    test_accuracies = []
    
    #Debug
    # train_correct = 0
    # train_total = 0
    # test_correct = 0
    # test_total = 0

    for client_results in all_client_results:
        # Sum correct predictions and total samples across all clients
        train_correct = client_results[:, 0, 0].sum()
        train_total = client_results[:, 0, 1].sum()
        test_correct = client_results[:, 1, 0].sum()
        test_total = client_results[:, 1, 1].sum()

        # Append average accuracy for each round
        train_accuracies.append(train_correct / train_total if train_total != 0 else 0)
        test_accuracies.append(test_correct / test_total if test_total != 0 else 0)

    #Debug
    # print("last train_correct: ", train_correct)
    # print("last train_total: ", train_total)
    # print("last test_correct: ", test_correct)
    # print("last test_total: ", test_total)
    # print("train_accuracies: ", np.array(train_accuracies), np.array(train_accuracies).shape)
    # print("test_accuracies: ", np.array(test_accuracies), np.array(test_accuracies).shape)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.plot(
        all_eval_rounds, 
        100 * np.array(train_accuracies), 
        linewidth=5.0, 
        label="Train Acc.")
    
    ax.plot(
        all_eval_rounds, 
        100 * np.array(test_accuracies), 
        linewidth=5.0, 
        linestyle="dashed", 
        label="Test Acc.")

    ax.grid(True, linewidth=2)

    ax.set_xlabel("Round", fontsize=30)
    ax.set_ylabel("Accuracy (%)", fontsize=30)
    ax.tick_params(axis='both', labelsize=20)
    
    ax.legend(fontsize=20)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


# TODO: Check if this plot makes sense --> CAUTION: Check if the modified code is compatible with the method plot_hetero_effect where plot_capacity_effect is used
def plot_capacity_effect(ax, results_dir, save_path=None, label=None):
    """plot the effect of the interpolation parameter on the test accuracy

    :param ax:

    :param results_dir: directory storing the experiment results

    :param save_path: directory to save the plot, default is `None`, if not provided the plot is not saved

    :param label: label of the plot, default is None

    """
    all_scores = np.load(os.path.join(results_dir, "all_scores.npy"))
    n_val_samples = np.load(os.path.join(results_dir, "n_val_samples.npy"))
    weights_grid = np.load(os.path.join(results_dir, "weights_grid.npy"))
    capacities_grid = np.load(os.path.join(results_dir, "capacities_grid.npy"))

    average_scores = np.nan_to_num(all_scores).sum(axis=0) / n_val_samples.sum()

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

        ax.set_ylabel("Accuracy (%)", fontsize=30)
        ax.set_xlabel("Capacity", fontsize=30)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xticks(np.arange(0, 1. + 1e-6, 0.1))

        ax.legend(fontsize=20)

        plt.savefig(save_path, bbox_inches='tight')


def plot_weight_effect(results_dir, save_path=None):
    """

    :param results_dir:
    :param save_path:
    :return:

    """
    all_scores = np.load(os.path.join(results_dir, "all_scores.npy"))
    n_train_samples = np.load(os.path.join(results_dir, "n_train_samples.npy"))
    n_val_samples = np.load(os.path.join(results_dir, "n_val_samples.npy"))
    weights_grid = np.load(os.path.join(results_dir, "weights_grid.npy"))
    capacities_grid = np.load(os.path.join(results_dir, "capacities_grid.npy"))

    #Debug - Inspect optimal hyperparameters and average test accuracy
    accuracies = (np.nan_to_num(all_scores).sum(axis=0) / n_val_samples.sum()) * 100
    best_acc = np.max(accuracies)
    best_index = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    best_weight = weights_grid[best_index[0]]
    best_capacity = capacities_grid[best_index[1]]
    print(f"Optimal weight: {best_weight}, Optimal Capacity: {best_capacity}")

    all_test_scores = np.load(os.path.join(results_dir, "all_test_scores.npy"))
    n_test_samples = np.load(os.path.join(results_dir, "n_test_samples.npy"))
    average_test_accuracy = np.sum(all_test_scores) / np.sum(n_test_samples) * 100
    individual_accuracies = (all_test_scores / n_test_samples) * 100
    mean = np.mean(individual_accuracies)
    std = np.std(individual_accuracies)
    print(f'Average Test Accuracy: {average_test_accuracy:.4f}')
    print(f'Mean and Standard Deviation of Test Accuracies across Clients: Mean: {mean:.4f}%, Std: {std:.4f}%')

    average_scores = np.nan_to_num(all_scores).sum(axis=0) / n_val_samples.sum()
    # The capacity is normalized w.r.t. the initial size of the client’s (train) dataset partition (like in kNN-per) 
    average_n_train_samples = np.mean(n_train_samples)

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

    ax.set_ylabel("Test accuracy (%)", fontsize=30)
    ax.set_xlabel(r"$\lambda$", fontsize=30)
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(fontsize=20)

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

        ax.set_ylabel("Accuracy (%)", fontsize=30)
        ax.set_xlabel("Capacity", fontsize=30)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xticks(np.arange(0, 1. + 1e-6, 0.1))

        ax.legend(fontsize=20)

        plt.savefig(save_path, bbox_inches='tight')

    else:
        plt.show()

    plt.close()
