#!/bin/bash
# -----------------------------------------------------------------------------
# Train the 2-arm MMIDAS model on the snRNA (Dbh) data across 5 random seeds.
#
# Pipeline position:
#   notebooks/00a_prepare_data_for_mmidas.ipynb   (prepare data/snRNA_BN_norm1.h5ad)
#   scripts/train_aug_mixvae.sh                   (train the augmenter)  <-- run first
#   scripts/train_mixvae_slurmarray.sh            (THIS: train 5 seeds)
#   notebooks/00b_two_arms_multiseed.ipynb        (load results, make figures)
#
# Outputs are written under the `saving_path_randomseed` location defined in
# pyproject.toml; 00b reads the trained models from there.
#
# NOTE: the #SBATCH resource directives below are specific to the Allen
# Institute `aind` SLURM partition (A100 GPUs). Adapt -p / --gpus / paths and
# the conda env name to your cluster. Log paths are relative to the submission
# directory; ensure ./logfiles exists (a .gitkeep is committed for this).
# Submit from the repo root:  sbatch scripts/train_mixvae_slurmarray.sh
# -----------------------------------------------------------------------------
#SBATCH -N1
#SBATCH --gpus=a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p aind
#SBATCH --time=40:00:00
#SBATCH --array=0-4
#SBATCH -o logfiles/seed_%A_%a.out
#SBATCH -e logfiles/seed_%A_%a.err
#SBATCH --job-name=mixvae_array

seeds=(1 2 3 4 5)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

hostname; date
start_time=$(date +%s)
echo "Job started at: $(date) with seed: $seed (task ID: $SLURM_ARRAY_TASK_ID)"

source activate mmidas
python train_mixvae.py --tau 0.1 --state_dim 3 --n_arm 2 --cuda --latent_dim 10 --seed $seed --augmentation

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job finished at: $(date)"
echo "Total runtime: $runtime seconds ($(($runtime / 60)) minutes)"
