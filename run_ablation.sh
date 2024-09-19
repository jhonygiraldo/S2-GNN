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
datasets=("activity" "isolet", "cancer_b", "cancer", "20news")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Run the Python script with the current GNN model and dataset
    python ablation_epsilon.py \
    --dataset "$dataset" \
    > "txt_ablations/${dataset}_knn.txt" &
done

# Wait for all background jobs to finish
wait
conda deactivate
echo 'python scripts have finished'