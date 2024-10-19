echo "=> generate data"

s_frac=0.3

rm -rf data/femnist/all_clients_data

python data/femnist/generate_data.py \
    --s_frac $s_frac \
    --test_frac 0.2 \
    --test_tasks_frac 0.2 \
    --seed 12345

# Get the current date and time in the format YYYY_MM_DD_HH_MM
current_time=$(date +"%Y_%m_%d_%H_%M")

echo "Train baseline non-parametric model | KNN_centroids (Ours) for DINOv2 embeddings"

for k in 3 5 7
do
  
  echo "Experiment with k = $k"

  results_dir="results/femnist/s_frac-${s_frac}/base_knn_centroids/${current_time}/n_neighbors_$k/"

  python eval_knn_centroids.py \
    femnist \
    random \
    --aggregator_type centralized_linear \
    --bz 128 \
    --n_neighbors $k \
    --n_clusters 10 \
    --capacities_grid_resolution 0.01 \
    --weights_grid_resolution 0.01 \
    --knn_weights gaussian_kernel \
    --gaussian_kernel_scale 1.0 \
    --device cuda \
    --results_dir $results_dir \
    --seed 1234  \
    --verbose 1

  echo "----------------------------------------------------"

done