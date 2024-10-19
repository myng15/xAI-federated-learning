"""
Process FEMNIST dataset, and splits it among clients
"""
import os
import time
import random
import argparse
import torch

from tqdm import tqdm

from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data/femnist", "intermediate", "data_as_tensor_by_writer")
TARGET_PATH = "data/femnist/all_clients_data/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--s_frac',
        help='fraction of data to be used; default is 0.3',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--test_frac',
        help='fraction in test set; default: 0.2;',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients  not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes;',
        type=int,
        default=12345,
        required=False
    )

    return parser.parse_args()


def save_task(dir_path, train_data, train_targets, test_data, test_targets):
    r"""
    save (`train_data`, `train_targets`) in {dir_path}/train.pt,
    (`val_data`, `val_targets`) in {dir_path}/val.pt
    and (`test_data`, `test_targets`) in {dir_path}/test.pt
    
    :param dir_path:
    :param train_data:
    :param train_targets:
    :param test_data:
    :param test_targets:
    """
    torch.save((train_data, train_targets), os.path.join(dir_path, "train.pt"))
    torch.save((test_data, test_targets), os.path.join(dir_path, "test.pt"))


def main():
    args = parse_args()

    rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    n_tasks = int(len(os.listdir(RAW_DATA_PATH)) * args.s_frac)
    file_names_list = os.listdir(RAW_DATA_PATH)
    rng.shuffle(file_names_list)

    file_names_list = file_names_list[:n_tasks]
    rng.shuffle(file_names_list)

    os.makedirs(os.path.join(TARGET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_PATH, "test"), exist_ok=True)

    print("generating data..")
    for idx, file_name in enumerate(tqdm(file_names_list)):
        if idx < int((1.0 - args.test_tasks_frac) * n_tasks):
            mode = "train"
        else:
            mode = "test"

        data, targets = torch.load(os.path.join(RAW_DATA_PATH, file_name), weights_only=True)
        train_data, test_data, train_targets, test_targets =\
            train_test_split(
                data,
                targets,
                train_size=(1.0 - args.test_frac),
                random_state=args.seed
            )

        save_path = os.path.join(TARGET_PATH, mode, f"task_{idx}")
        os.makedirs(save_path, exist_ok=True)

        save_task(
            dir_path=save_path,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )


if __name__ == "__main__":
    main()


