#!/bin/bash 
#SBATCH -J cwru_test
#SBATCH -p gpu-low
#SBATCH --gres=gpu:1
#SBATCH -N 1  
#SBATCH -n 4
#SBATCH -t 4300:00 
#SBATCH -o logs/cwru_uid_281_.out
#SBATCH -e logs/cwru_uid_281_.err


mkdir -p results
python3 main_eval.py \
    --logdir=logs --modeldir=models \
    --jobs=1 --gpus=1 --gpumem=0 \
    --match="cwru-255-wsmix_np-[0-9]*" --selection="best_target" \
    --output_file=results/results_best_target-cwru-255-wsmix_np.yaml





