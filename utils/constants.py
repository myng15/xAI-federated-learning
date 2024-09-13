import string


ALL_STRATEGIES = {
    "random"
}

ALL_MODELS = {
    "mobilenet"
}

LOADER_TYPE = {
    "synthetic": "tabular",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "femnist": "femnist",
    "shakespeare": "shakespeare"
}

EXTENSIONS = {
    "tabular": ".pkl",
    "cifar10": ".pkl",
    "cifar100": ".pkl",
    "femnist": ".pt",
    "shakespeare": ".txt"
}

N_CLASSES = {
    "synthetic": 1,
    "cifar10": 10,
    "cifar100": 100,
    "femnist": 62,
    "shakespeare": 100
}

EMBEDDING_DIM = {
    "cifar10": 768, #1280,
    "cifar100": 1280,
    "femnist": 1280,
    "shakespeare": 1024
}


LOCAL_HEAD_UPDATES = 10  # number of epochs for local heads used in FedRep

# NUM_WORKERS = os.cpu_count()  # number of workers used to load data and in GPClassifier
NUM_WORKERS = 1
