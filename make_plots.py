from utils.plots import *
from utils.args import PlotsArgumentsManager


if __name__ == "__main__":
    arguments_manager = PlotsArgumentsManager()
    arguments_manager.parse_arguments()

    results_dir = arguments_manager.args.results_dir
    # if "save_path" in arguments_manager.args:
    #     save_path = arguments_manager.args.save_path
    # else:
    #     save_path = None

    # Create save_dir based on results_dir
    save_dir = results_dir.replace("results", "plots", 1)
    os.makedirs(save_dir, exist_ok=True)

    if arguments_manager.args.plot_name == "fedavg_client_evaluation":
        plot_client_evaluation(
            results_dir=results_dir, 
            mode="train", 
            save_path=os.path.join(save_dir, "fedavg_all_client_results.png")
        )
        
        test_client_results_path = os.path.join(results_dir, "fedavg_all_test_client_results.npy")
        if os.path.exists(test_client_results_path):
            plot_client_evaluation(
            results_dir=results_dir, 
            mode="test", 
            save_path=os.path.join(save_dir, "fedavg_all_test_client_results.png")
        )
    
    elif arguments_manager.args.plot_name == "capacity_effect":
        fig, ax = plt.subplots(figsize=(12, 10))
        plot_capacity_effect(ax, results_dir=results_dir, save_path=os.path.join(save_dir, "capacity_effect.png"))

    elif arguments_manager.args.plot_name == "weight_effect":
        plot_weight_effect(results_dir=results_dir, save_path=os.path.join(save_dir, "weight_effect.png"))

    elif arguments_manager.args.plot_name == "hetero_effect":
        plot_hetero_effect(results_dir=results_dir, save_path=os.path.join(save_dir, "hetero_effect.png"))

    else:
        raise NotImplementedError(
            f"{arguments_manager.args.plot_name} is not a valid plot name, possible are:"
            "{'capacity_effect', 'weight_effect', 'hetero_effect', 'n_neighbors_effect'} "
        )
