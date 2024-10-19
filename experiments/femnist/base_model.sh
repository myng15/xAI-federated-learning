echo "=> generate data"

s_frac=0.1

rm -rf data/femnist/all_clients_data

python data/femnist/generate_data.py \
    --s_frac $s_frac \
    --test_frac 0.2 \
    --test_tasks_frac 0.2 \
    --seed 12345

# Get the current date and time in the format YYYY_MM_DD_HH_MM
current_time=$(date +"%Y_%m_%d_%H_%M")

echo "Train base model | FedAvg on linear classification layer for DINOv2 embeddings"

# Define logs and checkpoints directories with the current date and time
logs_dir="logs/femnist/s_frac-${s_frac}/fedavg_linear/${current_time}/"
chkpts_dir="chkpts/femnist/s_frac-${s_frac}/fedavg_linear/${current_time}/"
results_dir="results/femnist/s_frac-${s_frac}/fedavg_linear/${current_time}/"

echo "Run FedAvg lr=0.05"
python eval_fedavg.py \
    femnist \
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
