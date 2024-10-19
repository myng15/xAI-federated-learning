import os
import argparse
import pickle

import numpy as np

import sys
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(current))
# adding the root directory to the sys.path.
sys.path.append(root)

from sklearn.model_selection import train_test_split
from data.mnist.utils_data import *

RAW_DATA_PATH = os.path.join(root, "data/mnist/database/")  #"data/mnist/database/" 
PATH = os.path.join(root, "data/mnist/all_clients_data/") #"data/mnist/all_clients_data/"
N_CLASSES = 10


def save_data(data, path_):
    with open(path_, 'wb') as f:
        pickle.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Splits MNIST embeddings among n_tasks. Default usage splits dataset in an IID fashion. '
                    'Can be used with `pathological_split` or `by_labels_split` for different methods of non-IID splits.'
    )
    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True
    )
    parser.add_argument(
            "--split_method",
            help='method to be used to split data among n_tasks;'
                 ' possible are "iid", "by_labels_split" and "pathological_split";'
                 ' 1) "by_labels_split": the dataset will be split as follow:'
                 '  a) classes are grouped into `n_clusters`'
                 '  b) for each cluster `c`, samples are partitioned across clients using dirichlet distribution'
                 ' Inspired by "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440);'
                 ' 2) "pathological_split": the dataset will be split as follow:'
                 '  a) sort the data by label'
                 '  b) divide it into `n_clients * n_classes_per_client` shards, of equal size.'
                 '  c) assign each of the `n_clients` with `n_classes_per_client` shards'
                 ' Similar to "Communication-Efficient Learning of Deep Networks from Decentralized Data"'
                 ' __(https://arxiv.org/abs/1602.05629);'
                 'default is "iid"',
            type=str,
            default="iid"
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; ignored if `--by_labels_split` is not used; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar;'
             'ignored if `--by_labels_split` is not used; default is 0.5',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 1.0;',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating in the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345
    )

    return parser.parse_args()


def load_embeddings():
    # Load MNIST embeddings and labels from the .npz files
    train_data = np.load(os.path.join(RAW_DATA_PATH, 'train.npz'))
    test_data = np.load(os.path.join(RAW_DATA_PATH, 'test.npz'))

    train_embeddings = train_data['embeddings']
    train_labels = train_data['labels']
    test_embeddings = test_data['embeddings']
    test_labels = test_data['labels']

    # Combine train and test embeddings and labels into a single dataset
    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    return embeddings, labels


def main():
    args = parse_args()

    # Load the embeddings and labels
    embeddings, labels = load_embeddings()

    dataset = list(zip(embeddings, labels))  # Combine embeddings and labels into a list of tuples

    if args.split_method == "pathological_split":
        clients_indices =\
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )

    elif args.split_method == "by_labels_split":
        clients_indices = \
            by_labels_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed
            )
    else: #args.split_method == "iid"
        clients_indices = \
            iid_split(
                dataset=dataset,
                n_clients=args.n_tasks,
                frac=args.s_frac,
                seed=args.seed
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            indices = np.array(indices)
            train_indices = indices[indices < 60_000]  # Assuming first 60k are train embeddings
            test_indices = indices[indices >= 60_000]  # Assuming remaining are test embeddings

            if (len(train_indices) == 0) or (len(test_indices) == 0):
                continue
            

            # HANDLE ISSUE OF MISSING TEST LABELS IN TRAIN LABELS:
            # Get labels for the training and testing data
            train_labels = labels[train_indices]
            test_labels = labels[test_indices]

            # Filter out test samples whose labels are not present in the training set
            valid_test_indices = test_indices[np.isin(test_labels, train_labels)]


            # Save train and test indices for each client
            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(valid_test_indices, os.path.join(client_path, "test.pkl"))

            if len(valid_test_indices) < len(test_indices):
                print(f"Client task_{client_id}: Filtered out {len(test_indices) - len(valid_test_indices)} test samples due to missing train labels.")


if __name__ == "__main__":
    main()
