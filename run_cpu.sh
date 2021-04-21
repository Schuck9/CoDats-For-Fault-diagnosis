#!/bin/bash 
#SBATCH -J cwru_cpu
#SBATCH -p normal
#SBATCH -N 1  
#SBATCH -n 4
#SBATCH -t 4300:00 
#SBATCH -o logs/cwru_uid_51_48DriveEndFault_daws.out
#SBATCH -e logs/cwru_uid_51_48DriveEndFault_daws.err


python3 main.py \
    --logdir=logs --modeldir=models \
    --method=daws --dataset=cwru --sources=48DriveEndFault_0.021\
    --target=48DriveEndFault_0.021 --uid=51 --debugnum=0 --steps=30000 --gpumem=0 





