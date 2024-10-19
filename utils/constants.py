import string


ALL_STRATEGIES = {
    "random"
}

ALL_MODELS = {
    "mobilenet",
    "linear"
}

LOADER_TYPE = {
    "synthetic": "tabular",
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "mnist": "mnist",
    "femnist": "femnist",
    "organamnist": "organamnist",
    "dermamnist": "dermamnist",
    "shakespeare": "shakespeare"
}

EXTENSIONS = {
    "tabular": ".pkl",
    "cifar10": ".pkl",
    "cifar100": ".pkl",
    "mnist": ".pkl",
    "femnist": ".pt",
    "organamnist": ".pkl",
    "dermamnist": ".pkl",
    "shakespeare": ".txt"
}

N_CLASSES = {
    "synthetic": 1,
    "cifar10": 10,
    "cifar100": 100,
    "mnist": 10,
    "femnist": 62,
    "organamnist": 11,
    "dermamnist": 7,
    "shakespeare": 100
}

EMBEDDING_DIM = {
    "cifar10": 768, 
    "cifar100": 768, 
    "mnist": 768,
    "femnist": 768, 
    "organamnist": 768,
    "dermamnist": 768,
    "shakespeare": 1024
}


LOCAL_HEAD_UPDATES = 10  # number of epochs for local heads used in FedRep

# NUM_WORKERS = os.cpu_count()  # number of workers used to load data and in GPClassifier
NUM_WORKERS = 1
