#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH --time=24:00:00

source ~/.bashrc

conda activate mixrep2

python train_mixvae.py  --tau 0.1 --state_dim 2 --device 0 