#!/bin/bash

#SBATCH --output=logs/jobGPU%j.out
#SBATCH --error=logs/jobGPU%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=10
#SBATCH --partition=V100-32GB

set -x

cd /home/ids/jgiraldo/S2-GNN-dev-master
pwd

echo $CUDA_VISIBLE_DEVICES

eval "$(conda shell.bash hook)"
conda init bash
conda activate env_pyg_v2
echo 'Virtual environment activated'

# Define arrays for GNN models and datasets
GNN_models=("Cheby" "SGC", "GAT", "ClusterGCN", "SSobGNN", 
            "GCN", "SuperGAT", "GATv2", "Transformer")
datasets=("cancer_b", "cancer", "20news", "activity", "isolet")
grahp=("knn" "learned")

# Loop through each GNN model
for GNN in "${GNN_models[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Loop through each graph typeand
        for graph in "${graph[@]}"; do
            # Run the Python script with the current GNN model and dataset
            python compute_test_results.py \
            --dataset "$dataset" \
            --graph "$graph" \
            --GNN "$GNN" \
            > "txt_results/${dataset}_${graph}_${GNN}.txt" &
        done
    done
done

# Wait for all background jobs to finish
wait
conda deactivate
echo 'python scripts have finished'