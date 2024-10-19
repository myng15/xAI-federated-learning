echo "=> generate data"

alpha=0.3
split_method="by_labels_split"

rm -rf data/organamnist/all_clients_data

python data/organamnist/generate_data.py \
    --n_tasks 200 \
    --split_method $split_method \
    --n_components -1 \
    --alpha $alpha \
    --s_frac 1.0 \
    --test_tasks_frac 0.2 \
    --seed 12345

# Get the current date and time in the format YYYY_MM_DD_HH_MM
current_time=$(date +"%Y_%m_%d_%H_%M")

echo "Train base model | FedAvg on linear classification layer for DINOv2 embeddings"

# Define logs and checkpoints directories with the current date and time
logs_dir="logs/organamnist/${split_method}/alpha-${alpha}/fedavg_linear/${current_time}/"
chkpts_dir="chkpts/organamnist/${split_method}/alpha-${alpha}/fedavg_linear/${current_time}/"
results_dir="results/organamnist/${split_method}/alpha-${alpha}/fedavg_linear/${current_time}/"

echo "Run FedAvg lr=0.05"
python eval_fedavg.py \
    organamnist \
    --model_name linear \
    --aggregator_type centralized_linear \
    --bz 128 \
    --n_rounds 100 \
    --lr 0.05 \
    --lr_scheduler multi_step \
    --log_freq 10 \
    --eval_freq 10 \
    --device cuda \
    --optimizer sgd \
    --logs_dir $logs_dir \
    --chkpts_dir $chkpts_dir \
    --results_dir $results_dir \
    --seed 1234  \
    --verbose 1
