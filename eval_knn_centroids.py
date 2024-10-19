from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.args import TestArgumentsManager

def eval_knn_grid(client_, weights_grid_, capacities_grid_):
    client_results = np.zeros((len(weights_grid_), len(capacities_grid_)))

    for ii, capacity in enumerate(capacities_grid_):
        client_.capacity = capacity
        client_.clear_datastore()
        client_.build_datastore()

        for jj, weight in enumerate(weights_grid_):
            client_results[jj, ii] = client_.evaluate(weight, val_mode=True) * client_.n_val_samples

    return client_results

def eval_knn(client_, weight_, capacity_):
    # Set client to the optimal capacity
    client_.capacity = capacity_
    
    # Clear datastore and rebuild it with the chosen capacity
    client_.clear_datastore()
    client_.build_datastore()
    
    # Evaluate the client with the optimal weight (λ) and fixed capacity
    client_results = client_.evaluate(weight_, val_mode=False) * client_.n_test_samples
    return client_results

def run(arguments_manager_):

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args
    
    seed_everything(args_.seed)

    rng_seed = (args_.seed if (("seed" in args_) and (args_.seed >= 0)) else int(time.time()))
    rng = np.random.default_rng(seed=rng_seed)

    data_dir = get_data_dir(args_.experiment)

    weights_grid_ = np.arange(0, 1. + 1e-6, args_.weights_grid_resolution)
    capacities_grid_ = np.arange(0., 1. + 1e-6, args_.capacities_grid_resolution)

    all_scores_ = []
    n_train_samples_ = []
    n_val_samples_ = []

    print("===> Initializing clients...")
    _, train_loaders, test_loaders = get_loaders(
        type_=LOADER_TYPE[args_.experiment],
        aggregator_= args_.aggregator_type, #"cluster_centroids"
        data_dir=os.path.join(data_dir, "train"),
        batch_size=args_.bz,
        is_validation=False
    )

    num_clients = len(train_loaders)
    features_dimension = EMBEDDING_DIM[args_.experiment]
    aggregator = ClusterCentroidsAggregator(num_clients=num_clients, features_dimension=features_dimension)

    clients = []
    
    # Initialize a client object for every `train loader` and `test loader`
    for train_loader, test_loader in tqdm(
        zip(train_loaders, test_loaders), 
        total=num_clients
    ):
        # Perform train/validation split
        val_size = int(len(train_loader.dataset) * args_.val_frac)
        train_size = len(train_loader.dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size)

        if args_.verbose > 0:
            print(f"N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

        # PART 1: TUNE HYPERPARAMETERS USING VALIDATION SET
        print("===> Evaluation on validation set starts...")
        client = KNNClusterCentroidsClient(
            learner=None,
            train_iterator=train_loader,
            val_iterator=val_loader,
            test_iterator=test_loader,
            logger=None,
            k=args_.n_neighbors,
            n_clusters=args_.n_clusters,
            features_dimension=features_dimension,
            num_classes=N_CLASSES[args_.experiment],
            capacity=-1,
            strategy=args_.strategy,
            rng=rng,
            knn_weights=args_.knn_weights,
            gaussian_kernel_scale=args_.gaussian_kernel_scale
        )

        if client.n_train_samples == 0 or client.n_test_samples == 0:
            continue

        clients.append(client)

    # Phase 1: Clients send centroids to the aggregator
    all_centroids = []
    all_labels = []
    
    print("===> Clients send centroids to the aggregator...")
    
    for client in tqdm(clients):
        client.load_all_features_and_labels()
        centroids, labels = client.compute_local_centroids()
        all_centroids.append(centroids)
        all_labels.append(labels)

    # Phase 2: Aggregator aggregates the centroids
    print("===> Aggregator aggregates the centroids...")
    aggregator.aggregate_centroids(all_centroids, all_labels)

    # Phase 3: Clients receive relevant centroids and perform classification
    print("===> Clients receive relevant centroids and perform classification...")

    # Tuning λ and capacity using validation set
    best_weight = 0.0
    best_capacity = 0.0

    for client in tqdm(clients):
        relevant_centroids, relevant_labels = aggregator.send_relevant_centroids(np.unique(client.train_labels)) 
        client.integrate_global_data(relevant_centroids, relevant_labels)

        # Evaluation
        client_scores = eval_knn_grid(client, weights_grid_, capacities_grid_)
        n_train_samples_.append(client.n_train_samples)
        n_val_samples_.append(client.n_val_samples)
        all_scores_.append(client_scores)

    all_scores_ = np.array(all_scores_)
    n_train_samples_ = np.array(n_train_samples_)
    n_val_samples_ = np.array(n_val_samples_)

    # Calculate average test accuracy (across all clients) for each combination of weight and capacity and find the best combo
    accuracies = (np.nan_to_num(all_scores_).sum(axis=0) / n_val_samples_.sum()) * 100
    best_acc = np.max(accuracies)
    best_index = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    best_weight = weights_grid_[best_index[0]]
    best_capacity = capacities_grid_[best_index[1]]

    print(f"Optimal weight: {best_weight}, Optimal Capacity: {best_capacity}")

    # PART 2: EVALUATE ON TEST SET USING OPTIMAL HYPERPARAMETERS
    print("===> Evaluation on test set starts...")

    all_test_scores_ = []
    n_test_samples_ = []

    for client in tqdm(clients):
        test_score = eval_knn(client, best_weight, best_capacity)
        all_test_scores_.append(test_score)
        n_test_samples_.append(client.n_test_samples)
    
    all_test_scores_ = np.array(all_test_scores_) 
    n_test_samples_ = np.array(n_test_samples_)

    average_test_accuracy = np.sum(all_test_scores_) / np.sum(n_test_samples_) * 100
    individual_accuracies = (all_test_scores_ / n_test_samples_) * 100
    mean = np.mean(individual_accuracies)
    std = np.std(individual_accuracies)

    print(f'Average Test Accuracy: {average_test_accuracy:.4f}%')
    print(f'Mean and Standard Deviation of Test Accuracies across Clients: Mean: {mean:.4f}%, Std: {std:.4f}%')

    return all_scores_, n_train_samples_, n_val_samples_, weights_grid_, capacities_grid_, all_test_scores_, n_test_samples_


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TestArgumentsManager()
    arguments_manager.parse_arguments()

    all_scores, n_train_samples, n_val_samples, weights_grid, capacities_grid, all_test_scores, n_test_samples = run(arguments_manager)

    if "results_dir" in arguments_manager.args:
        results_dir = arguments_manager.args.results_dir
    else:
        results_dir = os.path.join("results", arguments_manager.args_to_string())

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "all_scores.npy"), all_scores)
    np.save(os.path.join(results_dir, "n_train_samples.npy"), n_train_samples)
    np.save(os.path.join(results_dir, "n_val_samples.npy"), n_val_samples)
    np.save(os.path.join(results_dir, "weights_grid.npy"), weights_grid)
    np.save(os.path.join(results_dir, "capacities_grid.npy"), capacities_grid)
    np.save(os.path.join(results_dir, "all_test_scores.npy"), all_test_scores)
    np.save(os.path.join(results_dir, "n_test_samples.npy"), n_test_samples)
