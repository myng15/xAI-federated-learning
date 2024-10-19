echo "=> generate data"

alpha=0.5
split_method="by_labels_split"

rm -rf data/dermamnist/all_clients_data

python data/dermamnist/generate_data.py \
    --n_tasks 100 \
    --split_method $split_method \
    --n_components -1 \
    --alpha $alpha \
    --s_frac 1.0 \
    --test_tasks_frac 0.2 \
    --seed 12345

# Get the current date and time in the format YYYY_MM_DD_HH_MM
current_time=$(date +"%Y_%m_%d_%H_%M")

echo "Train baseline non-parametric model | KNN_centroids (Ours) for DINOv2 embeddings"

for k in 3 5 7
do
  
  echo "Experiment with k = $k"

  results_dir="results/dermamnist/${split_method}/alpha-${alpha}/base_knn_centroids/${current_time}/n_neighbors_$k/"

  python eval_knn_centroids.py \
    dermamnist \
    random \
    --aggregator_type centralized_linear \
    --bz 128 \
    --n_neighbors $k \
    --n_clusters 5 \
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