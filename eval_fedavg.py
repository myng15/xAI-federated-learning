from utils.utils import *
from utils.constants import *
from utils.args import TrainArgumentsManager

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, data_dir, logs_dir, chkpts_dir):
    """
    initialize clients from data folders

    :param args_:
    :param data_dir: path to directory containing data folders
    :param logs_dir: directory to save the logs
    :param chkpts_dir: directory to save chkpts
    :return: List[Client]

    """
    os.makedirs(chkpts_dir, exist_ok=True)

    print("===> Initializing clients..")
    train_loaders, _, test_loaders = \
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            aggregator_=args_.aggregator_type, # aggregator_type == "centralized_linear", #added argument to accommodate embedding databases as input in non-parametric mode (parametric mode such as FedAvg requires raw image datasets as input)
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    num_clients = len(train_loaders)

    clients_ = []

    for task_id, (train_loader, test_loader) in \
            enumerate(tqdm(zip(train_loaders, test_loaders), total=num_clients)):

        if train_loader is None or test_loader is None:
            continue

        # Perform train/validation split
        val_size = int(len(train_loader.dataset) * args_.val_frac)
        train_size = len(train_loader.dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)

        if args_.verbose > 0:
            print(f"N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

        learner =\
            get_learner(
                name=args_.experiment,
                model_name=args_.model_name,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                input_dimension=EMBEDDING_DIM[args_.experiment], #args_.input_dimension,
                hidden_dimension=None, #args_.hidden_dimension,
                mu=args_.mu # TODO: maybe remove this if not used
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=args_.client_type,
            learner=learner,
            train_iterator=train_loader,
            val_iterator=val_loader,
            test_iterator=test_loader,
            logger=logger,
            local_steps=args_.local_steps,
            client_id=task_id,
            save_path=os.path.join(chkpts_dir, "task_{}.pt".format(task_id))
        )

        clients_.append(client)

    return clients_


def run(arguments_manager_):
    """

    :param arguments_manager_:
    :type arguments_manager_: ArgumentsManager

    """

    if not arguments_manager_.initialized:
        arguments_manager_.parse_arguments()

    args_ = arguments_manager_.args

    seed_everything(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", arguments_manager_.args_to_string()) # TODO: check args_to_string()

    if "chkpts_dir" in args_:
        chkpts_dir = args_.chkpts_dir
    else:
        chkpts_dir = os.path.join("chkpts", arguments_manager_.args_to_string()) # TODO: check args_to_string()

    print("==> Clients initialization starts...")
    clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "train"),
            logs_dir=os.path.join(logs_dir, "train"),
            chkpts_dir=os.path.join(chkpts_dir, "train")
        )

    print("==> Test Clients initialization starts...")
    test_clients = \
        init_clients(
            args_,
            data_dir=os.path.join(data_dir, "test"),
            logs_dir=os.path.join(logs_dir, "test"),
            chkpts_dir=os.path.join(chkpts_dir, "test")
        )

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learner = \
        get_learner(
            name=args_.experiment,
            model_name=args_.model_name, # model_name == "linear"
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            input_dimension=EMBEDDING_DIM[args_.experiment], #args_.input_dimension,
            hidden_dimension=None #args_.hidden_dimension
        )

    aggregator = \
        get_aggregator(
            aggregator_type=args_.aggregator_type, # aggregator_type == centralized_linear
            clients=clients,
            global_learner=global_learner,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )

    all_client_results_ = []
    all_test_client_results_ = []
    all_eval_rounds_ = []

    aggregator.write_logs()

    # Evaluate each client's performance before training
    if args_.eval_freq is not None and args_.eval_freq != 0:
        client_results, test_client_results = aggregator.evaluate()
        #print(f"Client evaluation results before training: Train clients: {client_results}, Test clients: {test_client_results}")

        all_client_results_.append(client_results)
        all_test_client_results_.append(test_client_results)
        all_eval_rounds_.append(0)

    print("Training..")
    for ii in tqdm(range(args_.n_rounds)):
        aggregator.mix()

        if (ii % args_.log_freq) == (args_.log_freq - 1):
            aggregator.save_state(chkpts_dir)
            aggregator.write_logs()

        # Optional: Evaluate each client's performance at intermediate rounds
        if args_.eval_freq is not None and args_.eval_freq != 0:
            if (ii % args_.eval_freq) == (args_.eval_freq - 1) or ii == (args_.n_rounds - 1):  # Evaluate at `eval_freq` rounds and the last round
                client_results, test_client_results = aggregator.evaluate()  
                #print(f"Client evaluation results at round {ii+1}: Train clients: {client_results}, Test: {test_client_results}")

                all_client_results_.append(client_results)
                all_test_client_results_.append(test_client_results)
                all_eval_rounds_.append(ii+1)

    aggregator.save_state(chkpts_dir)

    all_client_results_ = np.array(all_client_results_)
    all_test_client_results_ = np.array(all_test_client_results_)
    all_eval_rounds_ = np.array(all_eval_rounds_)
    
    return all_client_results_, all_test_client_results_, all_eval_rounds_


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arguments_manager = TrainArgumentsManager()
    arguments_manager.parse_arguments()

    all_client_results, all_test_client_results, all_eval_rounds = run(arguments_manager)

    if "results_dir" in arguments_manager.args:
        results_dir = arguments_manager.args.results_dir
    else:
        results_dir = os.path.join("results", arguments_manager.args_to_string())

    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "fedavg_all_eval_rounds.npy"), all_eval_rounds)
    np.save(os.path.join(results_dir, "fedavg_all_client_results.npy"), all_client_results)
    if all_test_client_results.shape[1] > 0:
        np.save(os.path.join(results_dir, "fedavg_all_test_client_results.npy"), all_test_client_results)

