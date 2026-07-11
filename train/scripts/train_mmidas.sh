#!/bin/bash
#SBATCH -N1
#SBATCH --gpus=a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p celltypes
#SBATCH --time=24:00:00

source ~/.bashrc

conda activate mmidas

python train_mmidas.py \
        --n_categories 15 \
        --state_dim 3 \
        --n_arm 2 \
        --temp 1 \
        --tau 0.1 \
        --beta 1 \
        --lam 1 \
        --lam_pc 1 \
        --latent_dim 10 \
        --n_epoch 10000 \
        --n_epoch_p 10000 \
        --min_con 0.99 \
        --max_prun_it 14 \
        --n_aug_smp 0 \
        --fc_dim 100 \
        --batch_size 256 \
        --lr 0.001 \
        --n_gene 0 \
        --p_drop 0.25 \
        --s_drop 0.0 \
        --n_run 2 \
        --n_prun_c 0 \
        --training_mode MSE \
        --seed 0 \
        --toml_file paths.toml \
        --cuda
    # Optional flags (off by default): --augmentation --ref_pc --hard