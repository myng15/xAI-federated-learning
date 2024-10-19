import os
import time
import random

from copy import deepcopy

from abc import ABC, abstractmethod

import torch

from utils.torch_utils import *

#from learners.learners_ensemble import *

from tqdm import tqdm

import numpy as np
import numpy.linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients

    test_clients

    n_clients:

    n_test_clients

    clients_weights:

    global_learner: List[Learner]

    model_dim: dimension if the used model

    device:

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients_per_round:

    sampled_clients:

    c_round: index of the current communication round

    global_train_logger:

    global_test_logger:

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    rng: random number generator

    Methods
    ----------
    __init__

    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        self.global_learner = global_learner
        self.model_dim = self.global_learner.model_dim
        self.device = self.global_learner.device

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)

        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32,
                device=self.device
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients_ids = list()
        self.sampled_clients = list()

        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger
        self.log_freq = log_freq
        self.verbose = verbose

        self.c_round = 0

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def toggle_client(self, client_id, mode):
        """
        toggle client at index `client_id`, if `mode=="train"`, `client_id` is selected in `self.clients`,
        otherwise it is selected in `self.test_clients`.

        :param client_id: (int)
        :param mode: possible are "train" and "test"

        """
        pass

    def toggle_clients(self):
        for client_id in range(self.n_clients):
            self.toggle_client(client_id, mode="train")

    def toggle_sampled_clients(self):
        for client_id in self.sampled_clients_ids:
            self.toggle_client(client_id, mode="train")

    def toggle_test_clients(self):
        for client_id in range(self.n_test_clients):
            self.toggle_client(client_id, mode="test")

    def write_logs(self):
        self.toggle_test_clients()

        for global_logger, clients, mode in [
            (self.global_train_logger, self.clients, "train"),
            (self.global_test_logger, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            train_accuracies = []
            test_accuracies = []

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")
                    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.4f}%|", end="")
                    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.4f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            # Calculate unweighted mean and standard deviations for train and test accuracies
            train_acc_mean, train_acc_std = np.mean(train_accuracies), np.std(train_accuracies)
            test_acc_mean, test_acc_std = np.mean(test_accuracies), np.std(test_accuracies)

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.4f} | Train Acc: {global_train_acc * 100:.4f}% |", end="")
                print(f"Test Loss: {global_test_loss:.4f} | Test Acc: {global_test_acc * 100:.4f}% |")
                print(f"Train Acc Mean/Std: {train_acc_mean * 100:.4f}%/{train_acc_std * 100:.4f}% | Test Acc Mean/Std: {test_acc_mean * 100:.4f}%/{test_acc_std * 100:.4f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    # TODO: currently not used, check if useful somewhere
    def evaluate(self):
        """
        evaluate the aggregator, returns the performance of every client in the aggregator

        :return
            clients_results: (np.array of size (self.n_clients, 2, 2))
                number of correct predictions and total number of samples per client both for train part and test part
            test_client_results: (np.array of size (self.n_test_clients))
                number of correct predictions and total number of samples per client both for train part and test part

        """

        clients_results = []
        test_client_results = []

        for results, clients, mode in [
            (clients_results, self.clients, "train"),
            (test_client_results, self.test_clients, "test")
        ]:
            if len(clients) == 0:
                continue

            print(f"evaluate {mode} clients..")
            for client_id, client in enumerate(tqdm(clients)):
                if not client.is_ready():
                    self.toggle_client(client_id, mode=mode)

                _, train_acc, _, test_acc = client.write_logs()

                results.append([
                    [train_acc * client.n_train_samples, client.n_train_samples],
                    [test_acc * client.n_test_samples, client.n_test_samples]
                ])

                client.free_memory()

        return np.array(clients_results, dtype=np.uint16), np.array(test_client_results, dtype=np.uint16)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:

        """
        save_path = os.path.join(dir_path, "global.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

        for client_id, client in enumerate(self.clients):
            self.toggle_client(client_id, mode="train")
            client.save_state()
            client.free_memory()

    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))
        for client_id, client in self.clients:
            self.toggle_client(client_id, mode="train")
            client.load_state()
            client.free_memory()

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients_ids = \
                self.rng.choices(
                    population=range(self.n_clients),
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients_ids = self.rng.sample(range(self.n_clients), k=self.n_clients_per_round)

        self.sampled_clients = [self.clients[id_] for id_ in self.sampled_clients_ids]


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """
    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.step()

        learners = [client.learner for client in self.sampled_clients]

        average_learners(
            learners=learners,
            target_learner=self.global_learner,
            weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=True,
            average_gradients=False
        )

        for client in self.clients:
            copy_model(client.learner.model, self.global_learner.model)

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.is_ready():
            copy_model(client.learner.model, self.global_learner.model)
        else:
            client.learner = deepcopy(self.global_learner)

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(
                self.global_learner.model.parameters()
            )

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`.

        :param dir_path:

        """
        save_path = os.path.join(dir_path, f"global_{self.c_round}.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

    def load_state(self, dir_path):
        """
        load the state of the aggregator

        :param dir_path:

        """
        chkpts_path = os.path.join(dir_path, f"global_{self.c_round}.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpts_path))


class ClusterCentroidsAggregator:
    def __init__(self, num_clients, features_dimension):
        self.num_clients = num_clients
        self.global_centroids = []
        self.global_labels = []
        self.features_dimension = features_dimension

    def aggregate_centroids(self, all_centroids, all_labels):
        # Store all received centroids and labels
        self.global_centroids = np.vstack(all_centroids)
        self.global_labels = np.concatenate(all_labels)

    def send_relevant_centroids(self, client_labels):
        # Send back centroids relevant to the client's local labels
        relevant_indices = [i for i, label in enumerate(self.global_labels) if label in client_labels]
        return self.global_centroids[relevant_indices], self.global_labels[relevant_indices]
    

class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        pass

