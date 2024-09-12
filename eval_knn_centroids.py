from utils.utils import *
from utils.args import TestArgumentsManager

def eval_knn_grid(client_, weights_grid_, capacities_grid_):
    client_results = np.zeros((len(weights_grid_), len(capacities_grid_)))

    for ii, capacity in enumerate(capacities_grid_):
        client_.capacity = capacity
        #Debug
        print(f"{ii}'s capacity: {capacity}")

        client_.clear_datastore()
        client_.build_datastore()

        for jj, weight in enumerate(weights_grid_):
            #Debug
            print(f"{jj}'s weight: {weight}")

            client_results[jj, ii] = client_.evaluate(weight) * client_.n_test_samples

    return client_results


def run(arguments_manager_):

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args

    rng_seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed=rng_seed)

    data_dir = get_data_dir(args_.experiment)

    weights_grid_ = np.arange(0, 1. + 1e-6, args_.weights_grid_resolution)
    capacities_grid_ = np.arange(0., 1. + 1e-6, args_.capacities_grid_resolution)

    #Debug
    #print("weights_grid: ", weights_grid_)
    #print("capacities_grid_: ", capacities_grid_)

    all_scores_ = []
    n_train_samples_ = []
    n_test_samples_ = []

    _, train_loaders, test_loaders = get_loaders(
        type_=LOADER_TYPE[args_.experiment],
        aggregator_="",
        data_dir=os.path.join(data_dir, "train"),
        batch_size=args_.bz,
        is_validation=False
    )

    num_clients = len(train_loaders)
    features_dimension = 768 # EMBEDDING_DIM[args_.experiment]
    aggregator = ClusterCentroidsAggregator(num_clients=num_clients, features_dimension=features_dimension)

    clients = []
    
    # Initialize a client object for every `train loader` and `test loader`
    for train_loader, test_loader in tqdm(
        zip(train_loaders, test_loaders), 
        total=num_clients
    ):
        if args_.verbose > 0:
            print(f"N_Train: {len(train_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

        client = KNNClusterCentroidsClient(
            learner=None,
            train_iterator=train_loader,
            test_iterator=test_loader,
            logger=None,
            k=args_.n_neighbors,
            n_clusters=args_.n_clusters,
            features_dimension=features_dimension,
            num_classes=N_CLASSES[args_.experiment],
            capacity=-1,
            strategy=args_.strategy,
            rng=rng
        )

        if client.n_train_samples == 0 or client.n_test_samples == 0:
            continue

        clients.append(client)

    # Phase 1: Clients send centroids to the aggregator
    all_centroids = []
    all_labels = []
    
    print("Clients send centroids to the aggregator...")
    
    for client in tqdm(clients):
        client.load_all_features_and_labels()
        centroids, labels = client.compute_local_centroids()
        all_centroids.append(centroids)
        all_labels.append(labels)

        #Debug:
        #print("centroids: ", centroids, len(centroids), "; labels: ", labels, len(labels))

    # Phase 2: Aggregator aggregates the centroids
    print("Aggregator aggregates the centroids...")
    aggregator.aggregate_centroids(all_centroids, all_labels)

    # Phase 3: Clients receive relevant centroids and perform classification
    print("Clients receive relevant centroids and perform classification...")

    for client in tqdm(clients):
        #Debug:
        print("Client's train labels: ", np.unique(client.train_labels), client.train_labels.shape, "; Client's test labels: ", np.unique(client.test_labels), client.test_labels.shape)
        
        relevant_centroids, relevant_labels = aggregator.send_relevant_centroids(client.train_labels)
        client.integrate_global_data(relevant_centroids, relevant_labels)

        # Evaluation
        client_scores = eval_knn_grid(client, weights_grid_, capacities_grid_)
        n_train_samples_.append(client.n_train_samples)
        n_test_samples_.append(client.n_test_samples)
        all_scores_.append(client_scores)

    all_scores_ = np.array(all_scores_)
    n_test_samples_ = np.array(n_test_samples_)
    n_train_samples_ = np.array(n_train_samples_)

    return all_scores_, n_train_samples_, n_test_samples_, weights_grid_, capacities_grid_


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TestArgumentsManager()
    arguments_manager.parse_arguments()

    all_scores, n_train_samples, n_test_samples, weights_grid, capacities_grid = run(arguments_manager)

    if "results_dir" in arguments_manager.args:
        results_dir = arguments_manager.args.results_dir
    else:
        results_dir = os.path.join("results", arguments_manager.args_to_string())

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "all_scores.npy"), all_scores)
    np.save(os.path.join(results_dir, "n_train_samples.npy"), n_train_samples)
    np.save(os.path.join(results_dir, "n_test_samples.npy"), n_test_samples)
    np.save(os.path.join(results_dir, "weights_grid.npy"), weights_grid)
    np.save(os.path.join(results_dir, "capacities_grid.npy"), capacities_grid)
