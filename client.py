import numpy as np

from datastore import *
from utils.torch_utils import *
from utils.constants import *

from copy import deepcopy

from faiss import IndexFlatL2
from sklearn.cluster import KMeans
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Client(object):
    r"""
    Implements one client

    Attributes
    ----------
    learner
    train_iterator
    val_iterator
    test_iterator
    n_train_samples
    n_test_samples
    local_steps
    logger
    counter
    __save_path
    __id

    Methods
    ----------
    __init__
    step
    write_logs
    update_tuned_learners

    """
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            save_path=None,
            id_=None,
            *args,
            **kwargs
    ):
        """

        :param learner:
        :param train_iterator:
        :param val_iterator:
        :param test_iterator:
        :param logger:
        :param local_steps:
        :param save_path:

        """
        self.learner = learner

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.local_steps = local_steps

        self.save_path = save_path

        self.id = -1
        if id_ is not None:
            self.id = id_

        self.counter = 0
        self.logger = logger

    def is_ready(self):
        return self.learner.is_ready

    def step(self, *args, **kwargs):
        self.counter += 1

        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=self.local_steps,
        )

    def write_logs(self):

        train_loss, train_acc = self.learner.evaluate_iterator(self.val_iterator)
        test_loss, test_acc = self.learner.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def save_state(self, path=None):
        """

        :param path: expected to be a `.pt` file

        """
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not saved", RuntimeWarning)
                return
            else:
                self.learner.save_checkpoint(self.save_path)
                return

        self.learner.save_checkpoint(path)

    def load_state(self, path=None):
        if path is None:
            if self.save_path is None:
                warnings.warn("client state was not loaded", RuntimeWarning)
                return
            else:
                self.learner.load_checkpoint(self.save_path)
                return

        self.learner.load_checkpoint(path)

    def free_memory(self):
        self.learner.free_memory()



class KNNRelativeSimilaritiesClient(Client):
    """

    Attributes
    ----------
    model:
    features_dimension:
    num_classes:
    train_loader:
    test_loader:
    n_train_samples:
    n_test_samples:
    local_steps:
    logger:
    binary_classification_flag:
    counter:
    capacity: datastore capacity of the client
    strategy: strategy to select samples to keep on the datastore
    rng (numpy.random._generator.Generator):
    datastore (datastore.DataStore):
    datastore_flag (bool):
    features_dimension (int):
    num_classes (int):
    train_features: (n_train_samples x features_dimension)
    test_features: (n_train_samples x features_dimension)
    model_outputs: (n_test_samples x num_classes)
    model_outputs_flag (bool):
    knn_outputs:
    knn_outputs_flag (bool)
    interpolate_logits (bool): if selected logits are interpolated instead of probabilities

    Methods
    -------
    __init__

    build

    compute_features_and_model_outputs

    build_datastore

    gather_knn_outputs

    evaluate

    clear_datastore

    """

    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            k,
            interpolate_logits,
            features_dimension,
            num_classes,
            capacity,
            strategy,
            rng,
            *args,
            **kwargs
    ):
        """

        :param learner:
        :param train_iterator:
        :param val_iterator:
        :param test_iterator:
        :param logger:
        :param local_steps:
        :param k:
        :param features_dimension:
        :param num_classes:
        :param capacity:
        :param strategy:
        :param rng:

        """
        super(KNNRelativeSimilaritiesClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            *args,
            **kwargs
        )

        self.k = k
        self.interpolate_logits = interpolate_logits

        self.model = self.learner.model
        self.features_dimension = features_dimension
        self.num_classes = num_classes

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(train_iterator.dataset)
        self.n_test_samples = len(test_iterator.dataset)

        self.capacity = capacity
        self.strategy = strategy
        self.rng = rng
        self.device = self.learner.device

        self.model = self.model.to(self.device)
        self.model.eval()

        self.datastore = DataStore(self.capacity, self.strategy, self.features_dimension, self.rng)
        self.datastore_flag = False

        self.train_features = np.zeros(shape=(self.n_train_samples, self.features_dimension), dtype=np.float32)
        self.train_labels = np.zeros(shape=self.n_train_samples, dtype=np.float32)
        self.test_features = np.zeros(shape=(self.n_test_samples, self.features_dimension), dtype=np.float32)
        self.test_labels = np.zeros(shape=self.n_test_samples, dtype=np.float32)

        self.train_model_outputs = np.zeros(shape=(self.n_train_samples, self.num_classes), dtype=np.float32)
        self.train_model_outputs_flag = False

        self.test_model_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.test_model_outputs_flag = False

        self.train_knn_outputs = np.zeros(shape=(self.n_train_samples, self.num_classes), dtype=np.float32)
        self.train_knn_outputs_flag = False

        self.test_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.test_knn_outputs_flag = False

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = int(k)

    @property
    def capacity(self):
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity):
        if 0 <= capacity <= 1 and isinstance(capacity, float):
            capacity = int(capacity * self.n_train_samples)
        else:
            capacity = int(capacity)

        if capacity < 0:
            capacity = self.n_train_samples

        self.__capacity = capacity

    def step(self):
        pass
    
    def build_datastore(self):
        self.datastore_flag = True
        
        self.datastore.build(self.train_features, self.train_labels)

    def gather_knn_outputs(self, mode="test", scale=1.):
        """
        computes the k-NN predictions

        :param mode: possible are "train" and "test", default is "test"
        :param scale: scale of the gaussian kernel, default is 1.0
        """
        if self.capacity <= 0:
            warnings.warn("trying to gather knn outputs with empty datastore", RuntimeWarning)
            return

        assert self.datastore_flag, "Should build datastore before computing knn outputs!"

        if mode == "train":
            features = self.train_features
            self.train_knn_outputs_flag = True
        else:
            features = self.test_features
            self.test_knn_outputs_flag = True

        distances, indices = self.datastore.index.search(features, self.k)
        similarities = np.exp(-distances / (self.features_dimension * scale))
        neighbors_labels = self.datastore.labels[indices]

        masks = np.zeros(((self.num_classes,) + similarities.shape))
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        if mode == "train":
            self.train_knn_outputs = outputs.T
        else:
            self.test_knn_outputs = outputs.T

    def evaluate(self, weight, mode="test"):
        """
        evaluates the client for a given weight parameter

        :param weight: float in [0, 1]
        :param mode: possible are "train" and "test", default is "test"

        :return:
            accuracy score

        """
        if mode == "train":
            flag = self.train_knn_outputs_flag
            knn_outputs = self.train_knn_outputs
            model_outputs = self.train_model_outputs
            labels = self.train_labels

        else:
            flag = self.test_knn_outputs_flag
            knn_outputs = self.test_knn_outputs
            model_outputs = self.test_model_outputs
            labels = self.test_labels

        if flag:
            outputs = weight * knn_outputs + (1 - weight) * model_outputs
        else:
            warnings.warn("evaluation is done only with model outputs, datastore is empty", RuntimeWarning)
            outputs = model_outputs

        predictions = np.argmax(outputs, axis=1)

        correct = (labels == predictions).sum()
        total = len(labels)

        if total == 0:
            acc = 1
        else:
            acc = correct / total

        return acc

    def clear_datastore(self):
        """
        clears `datastore`

        """
        self.datastore.clear()
        self.datastore.capacity = self.capacity

        self.datastore_flag = False
        self.train_knn_outputs_flag = False
        self.test_knn_outputs_flag = False


# TODO: Create a super class for all kNN-based client classes

class KNNClusterCentroidsClient(Client):
    def __init__(
            self, 
            learner,
            train_iterator, 
            test_iterator, 
            logger, 
            k,
            n_clusters, 
            features_dimension, 
            num_classes,
            capacity, 
            strategy, 
            rng, 
            *args, 
            **kwargs
    ):
        super(KNNClusterCentroidsClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=None,  # Assuming validation iterator is not needed here
            test_iterator=test_iterator,
            logger=logger,
            local_steps=None,  # Adjust if local steps are needed
            *args, 
            **kwargs
        )

        self.k = k

        self.n_clusters = n_clusters
        self.features_dimension = features_dimension
        self.num_classes = num_classes

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(train_iterator.dataset)
        self.n_test_samples = len(test_iterator.dataset)

        self.capacity = capacity
        self.strategy = strategy
        self.rng = rng

        # Initialize FAISS index for global centroids
        self.faiss_index = IndexFlatL2(self.features_dimension)
        self.global_labels = np.array([], dtype=np.int64)

        # Initialize local datastore for embeddings and labels
        self.datastore = DataStore(self.capacity, self.strategy, self.features_dimension, self.rng)
        self.datastore_flag = False

        # Initialize local training data
        self.train_features = np.zeros(shape=(self.n_train_samples, self.features_dimension), dtype=np.float32)
        self.train_labels = np.zeros(shape=self.n_train_samples, dtype=np.int64)
        self.test_features = np.zeros(shape=(self.n_test_samples, self.features_dimension), dtype=np.float32)
        self.test_labels = np.zeros(shape=self.n_test_samples, dtype=np.int64)

        self.local_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.local_knn_outputs_flag = False

        self.glocal_knn_outputs = np.zeros(shape=(self.n_test_samples, self.num_classes), dtype=np.float32)
        self.glocal_knn_outputs_flag = False

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = int(k)

    @property
    def capacity(self):
        return self.__capacity

    @capacity.setter
    def capacity(self, capacity):
        if 0 <= capacity <= 1 and isinstance(capacity, float):
            capacity = int(capacity * self.n_train_samples)
        else:
            capacity = int(capacity)

        if capacity < 0:
            capacity = self.n_train_samples

        self.__capacity = capacity

    def load_all_features_and_labels(self):
        start_index = 0
        for batch_features, batch_labels, _ in self.train_iterator:
            batch_size = batch_features.shape[0]
            end_index = start_index + batch_size
            
            # Copy data from PyTorch tensors to numpy arrays
            self.train_features[start_index:end_index] = batch_features.numpy()
            self.train_labels[start_index:end_index] = batch_labels.numpy()

            start_index += batch_size

        start_index = 0
        for batch_features, batch_labels, _ in self.test_iterator:
            batch_size = batch_features.shape[0]
            end_index = start_index + batch_size
            
            # Copy data from PyTorch tensors to numpy arrays
            self.test_features[start_index:end_index] = batch_features.numpy()
            self.test_labels[start_index:end_index] = batch_labels.numpy()

            start_index += batch_size

    def build_datastore(self):
        self.datastore_flag = True

        #Debug
        print("self.datastore.capacity in build_datastore(): ", self.datastore.capacity)

        self.datastore.build(self.train_features, self.train_labels)

    def compute_local_centroids(self):
        #Debug
        #print("Train features for KMeans: ", "Shape: ", self.train_features.shape, " | ", self.train_features[0:10])

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1234)
        kmeans.fit(self.train_features)
        centroids = kmeans.cluster_centers_
        labels = [np.argmax(np.bincount(self.train_labels[kmeans.labels_ == i])) for i in range(self.n_clusters)]
        return centroids, labels

    def integrate_global_data(self, global_centroids, global_labels):
        #Debug
        #print("global_centroids: ", type(global_centroids), global_centroids.shape)
        #print("global_labels: ", type(global_labels), global_labels.shape)

        #print("self.global_labels before: ", type(self.global_labels), self.global_labels.shape)

        self.faiss_index = IndexFlatL2(self.features_dimension)
        self.faiss_index.add(global_centroids)
        self.global_labels = np.concatenate([self.global_labels, global_labels])

        #print("self.global_labels after: ", type(self.global_labels), self.global_labels.shape)

    def compute_knn_outputs(self, features, scope="local", scale=1., method="gaussian_kernel"):
            """
            Computes k-NN outputs for given features.

            :param features: Features for which to compute k-NN outputs
            :param mode: 'local' or 'global', determines which datastore to use
            :param scale: Scale for Gaussian kernel
            :param inverse_distances: If True, use inverse distances as weights
            :param gaussian_kernel: If True, use Gaussian kernel as weights
            :return: k-NN outputs
            """
            if scope == "local":
                if self.capacity <= 0:
                    warnings.warn("trying to gather knn outputs with empty datastore", RuntimeWarning)
                    return

                assert self.datastore_flag, "Should build local datastore before computing knn outputs!"

                #Debug
                print("self.datastore.capacity in compute_knn_outputs(): ", self.datastore.capacity)
    
                distances, indices = self.datastore.index.search(features, self.k)
                if method == "inverse_distances":
                    knn_outputs = self._compute_weighted_outputs(distances, indices)
                elif method == "gaussian_kernel":
                    knn_outputs = self._compute_gaussian_kernel_outputs(distances, indices, scale)
                else:
                    raise ValueError("Invalid method. Use 'inverse_distances' or 'gaussian_kernel'.")
                
                self.local_knn_outputs_flag = True

            elif scope == "global":
                distances, indices = self.faiss_index.search(features, self.k)
                if method=="inverse_distances":
                    knn_outputs = self._compute_weighted_outputs(distances, indices, global_mode=True)
                elif method=="gaussian_kernel":
                    knn_outputs = self._compute_gaussian_kernel_outputs(distances, indices, scale, global_mode=True)
                else:
                    raise ValueError("Invalid method. Use 'inverse_distances' or 'gaussian_kernel'.")

                self.global_knn_outputs_flag = True

            else:
                raise ValueError("Scope must be 'local' or 'global'.")

            return knn_outputs

    def _compute_weighted_outputs(self, distances, indices, global_mode=False):
        if global_mode:
            labels = self.global_labels
        else:
            labels = self.datastore.labels

        # weights of each nearest neighbor in the train dataset to a certain test feature (w.r.t. their distances from this test feature)
        weights = 1. / (distances + 1e-8)  # Avoid division by zero 

        #Debug:
        #print("weights for computing weighted outputs: ", weights, weights.shape, "; - distances: ", distances.shape)

        knn_outputs = np.zeros((weights.shape[0], self.num_classes), dtype=np.float32) #knn_outputs: (n_test_samples,n_classes); weights: (n_test_samples, k)
        for i in range(weights.shape[0]): # for each test feature
            weighted_sum = np.zeros(self.num_classes, dtype=np.float32)

            # Debug:
            #print("weighted_sum: ", weighted_sum, weighted_sum.shape)     #weighted_sum: (10,)

            for j in range(weights.shape[1]): # for each neighbor of the test feature
                class_label = labels[indices[i, j]]
                weighted_sum[class_label] += weights[i, j]
            knn_outputs[i] = weighted_sum / weights[i].sum()

        #Debug
        #print("labels: ", labels.shape, labels)
        #print("knn_outputs: ", knn_outputs.shape)

        return knn_outputs

    def _compute_gaussian_kernel_outputs(self, distances, indices, scale, global_mode=False):
        if global_mode:
            labels = self.global_labels
        else:
            labels = self.datastore.labels

        similarities = np.exp(-distances / (self.features_dimension * scale))
        neighbors_labels = labels[indices]
        knn_outputs = np.zeros((similarities.shape[0], self.num_classes), dtype=np.float32)

        masks = np.zeros(((self.num_classes,) + similarities.shape), dtype=np.float32)
        for class_id in range(self.num_classes):
            masks[class_id] = neighbors_labels == class_id

        knn_outputs = (similarities * masks).sum(axis=2) / similarities.sum(axis=1)

        return knn_outputs.T

    def evaluate(self, weight):
        """
        Evaluates the client for a given weight parameter.

        :param weight: float in [0, 1]
        :return: accuracy score
        """
        self.local_knn_outputs = self.compute_knn_outputs(self.test_features, scope="local", scale=1., method="gaussian_kernel")
        self.global_knn_outputs = self.compute_knn_outputs(self.test_features, scope="global", scale=1., method="gaussian_kernel")
        #Debug
        #print("local_knn_outputs: ", "|", self.local_knn_outputs)
        #print("global_knn_outputs: ", "|", self.global_knn_outputs) 

        labels = self.test_labels
        #Debug
        #print("test labels: ", labels)

        if self.local_knn_outputs_flag and self.global_knn_outputs_flag:
            outputs = weight * self.local_knn_outputs + (1 - weight) * self.global_knn_outputs
        elif not self.local_knn_outputs_flag and self.global_knn_outputs_flag:
            warnings.warn("evaluation is done only with global outputs, local datastore is empty", RuntimeWarning)
            outputs = self.global_knn_outputs
        elif self.local_knn_outputs_flag and not self.global_knn_outputs_flag:
            warnings.warn("evaluation is done only with local outputs, global datastore is empty", RuntimeWarning)
            outputs = self.local_knn_outputs

        predictions = np.argmax(outputs, axis=1)
        correct = (labels == predictions).sum()
        total = len(labels)

        acc = correct / total if total > 0 else 1.0

        #Debug
        print("predictions: ", predictions.shape)
        print("correct: ", correct)

        return acc

    def clear_datastore(self):
        """
        clears local `datastore`

        """
        self.datastore.clear()
        self.datastore.capacity = self.capacity

        self.datastore_flag = False
        self.local_knn_outputs_flag = False
        self.global_knn_outputs_flag = False







