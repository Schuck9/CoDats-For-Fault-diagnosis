#!/bin/bash 
#SBATCH -J cwru_test
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1  
#SBATCH -n 4
#SBATCH -t 4300:00 
#SBATCH -o logs/ucihar_vrada.out
#SBATCH -e logs/ucihar_vrada.err


python3 main.py \
    --logdir=logs --modeldir=example-models \
    --method=vrada --dataset=ucihar --sources=1 \
    --target=2 --uid=11 --debugnum=0 --steps=500000 --gpumem=0 





