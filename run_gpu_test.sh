#!/bin/bash 
#SBATCH -J cwru_test
#SBATCH -p gpu
#SBATCH -N 1  
#SBATCH -n 1
#SBATCH -t 4300:00 
#SBATCH -o results/cwru_uid_42_12Drive_12Drive_rdann.out
#SBATCH -e results/cwru_uid_42_12Drive_12Drive_rdann.err



python3 main_eval.py \
    --logdir=logs --modeldir=models \
    --jobs=1 --gpus=1 --gpumem=0 \
    --match="cwru-41-rdann-[0-9]*" --selection="best_target" \
    --output_file=results/results_best_target-cwru-41-rdnn-0.yaml





