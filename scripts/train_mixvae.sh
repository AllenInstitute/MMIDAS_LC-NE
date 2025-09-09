#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p aind
#SBATCH --time=24:00:00
#SBATCH -o /home/shuonan.chen/scratch_shuonan/code/LC-NE-MixRep/logfiles/job_%j.out
#SBATCH -e /home/shuonan.chen/scratch_shuonan/code/LC-NE-MixRep/logfiles/job_%j.err
#SBATCH --job-name=mmidas
hostname; date

conda activate mmidas

python train_mixvae.py  --tau 0.1 --state_dim 2 --device 0 