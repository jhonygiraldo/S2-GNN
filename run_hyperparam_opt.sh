#!/bin/bash

#SBATCH --output=logs/jobGPU%j.out
#SBATCH --error=logs/jobGPU%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=6
#SBATCH --partition=V100

set -x

cd /home/ids/jgiraldo/S2-GNN-dev-master
pwd

echo $CUDA_VISIBLE_DEVICES

eval "$(conda shell.bash hook)"
conda init bash
conda activate env_pyg_v2
echo 'Virtual environment activated'

# Define arrays for GNN models and datasets
GNN_models=("Cheby" "SGC", "GAT", "ClusterGCN", "SSobGNN", "GCN", "SuperGAT", "GATv2", "Transformer")
datasets=("MUTAG" "ENZYMES" "PROTEINS")

# Loop through each GNN model
for GNN in "${GNN_models[@]}"; do
    # Loop through each dataset
    for dataset in "${datasets[@]}"; do
        # Run the Python script with the current GNN model and dataset
        python hyper_opt.py \
        --GNN "$GNN" \
        --dataset "$dataset" \
        > "txt_hyperparameters/${dataset}_${GNN}.txt" &
    done
done

# Wait for all background jobs to finish
wait
conda deactivate
echo 'python scripts have finished'