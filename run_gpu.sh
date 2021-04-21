#!/bin/bash 
#SBATCH -J cwru_test
#SBATCH -p gpu-high
#SBATCH --gres=gpu:1
#SBATCH -N 1  
#SBATCH -n 4
#SBATCH -t 4300:00 
#SBATCH -o logs/cwru_uid_303.out
#SBATCH -e logs/cwru_uid_303_.err


python3 main.py \
    --logdir=logs --modeldir=models \
    --method=wsmix_np --dataset=cwru --sources=12DriveEndFault_1797\
    --target=12DriveEndFault_1772 --uid=303   --debugnum=0  --steps=300000  --gpumem=0 




