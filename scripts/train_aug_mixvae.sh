#!/bin/bash
# -----------------------------------------------------------------------------
# Train the RNA augmenter on the snRNA (Dbh) data.
#
# Pipeline position (run this FIRST):
#   notebooks/00a_prepare_data_for_mmidas.ipynb   (prepare data/snRNA_BN_norm1.h5ad)
#   scripts/train_aug_mixvae.sh                   (THIS: train the augmenter)
#   scripts/train_mixvae_slurmarray.sh            (train the 2-arm model, 5 seeds)
#   notebooks/00b_two_arms_multiseed.ipynb        (load results, make figures)
#
# This writes a model file models/RNA_augmenter_Dbh_<timestamp>.pth. To use it
# downstream, set `augmenter_Dbh` in pyproject.toml to that filename. To force a
# fresh training run instead of loading an existing augmenter, set
# `augmenter_Dbh = "."` in pyproject.toml.
#
# NOTE: the #SBATCH resource directives below are specific to the Allen
# Institute `aind` SLURM partition (A100 GPUs). Adapt -p / --gpus / paths and
# the conda env name to your cluster. Log paths are relative to the submission
# directory; ensure ./logfiles_aug exists (a .gitkeep is committed for this).
# Submit from the repo root:  sbatch scripts/train_aug_mixvae.sh
# -----------------------------------------------------------------------------
#SBATCH -N1
#SBATCH --gpus=a100:1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -p aind
#SBATCH --time=44:00:00
#SBATCH -o logfiles_aug/job_%j.out
#SBATCH -e logfiles_aug/job_%j.err
#SBATCH --job-name=aug-vae

hostname; date
start_time=$(date +%s)
echo "Job started at: $(date)"

source activate mmidas
python train_aug_mixvae.py --tau 0.1 --state_dim 2 --n_arm 2 --cuda --latent_dim 10 --augmentation

end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job finished at: $(date)"
echo "Total runtime: $runtime seconds ($(($runtime / 60)) minutes)"
