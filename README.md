# MMIDAS LC-NE
![Python](https://img.shields.io/badge/python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-important?logo=nvidia)
![OS](https://img.shields.io/badge/Linux-Ubuntu-orange?logo=linux)


Learning mixture-based representations of locus coeruleus noradrenergic (**LC-NE**) neurons with **MMIDAS** (Mixture Model Inference with Discrete-coupled AutoencoderS).

This repository applies MMIDAS to single-cell/single-nucleus transcriptomic data from LC-NE neurons in order to jointly infer discrete cell types and the continuous, type-specific variability within them.

---

## Method

The analysis is built on [**MMIDAS**]((https://www.nature.com/articles/s43588-024-00683-8)), a generalized, unsupervised mixture variational model with a multi-armed deep neural network that jointly infers **discrete cell types** and **continuous type-specific variability** from single-cell datasets (uni- or multi-modal).

Here, MMIDAS is used to learn a mixture-based representation of LC-NE neurons, where:

- a categorical (discrete) latent variable captures cell-type identity, 
- a continuous state latent variable captures within-type variability.

Two coupled autoencoder *arms* are trained on augmented views of the data (produced by a VAE-GAN augmenter) and encouraged to reach a consensus assignment, which stabilizes the inferred categories and enables an automatic pruning of redundant types.

The reference implementation of MMIDAS is maintained in a separate repository:

> **MMIDAS** — https://github.com/AllenInstitute/MMIDAS


---

## Installation

### 1. Create a Conda environment

Create a new Conda environment and activate it:

```bash
conda create -n mmidas python=3.9
conda activate mmidas
```

### 2. Clone the repository

Clone the repository and change into it, then install the required packages listed in `requirements.txt`:

```bash
cd <directory where you want to place the repo>
git clone https://github.com/AllenInstitute/LC-NE-MixRep
cd LC-NE-MixRep
python -m pip install -r requirements.txt
```

### 3. Install PyTorch

PyTorch is not pinned in `requirements.txt` so that you can match it to your hardware.

**CPU only:**

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**GPU (CUDA):** PyTorch is compatible with NVIDIA GPUs through CUDA. Use `nvidia-smi` to find your installed CUDA version, then pick the matching build from the [PyTorch install page](https://pytorch.org/get-started/previous-versions/). For example, for CUDA 12.4:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 4. Install the package (editable mode)

```bash
python -m pip install -e .
```

### Docker (optional)

Instead of a Conda environment you can build a Docker image (Ubuntu + PyTorch + optional CUDA).

<details>
<summary>Show Docker instructions</summary>

Create a `Dockerfile`:

```dockerfile
# Use an image with CUDA pre-installed (only if you use GPU devices!)
FROM nvidia/cuda:12.4-cudnn8-runtime-ubuntu20.04

# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev

# Install python packages
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run Python
CMD ["python"]
```

Build the image:

```bash
docker build -t mmidas .
```

Run the container interactively:

```bash
docker run -it mmidas
```

Run a specific script (mounting the current directory):

```bash
docker run -v $(pwd):/app mmidas python src/train_middas.py
```

</details>

---

## Configuration

Paths, data files, and pretrained models are declared in a project **`.toml`** file (default: `paths.toml`). Every training script accepts a `--toml_file` argument so you can point it at a different config.

Example `paths.toml`:

```toml
# -----------------------------------------------------------------------------
[project]
name = "LC-NE-MixRep"
version = "0.1.1"
requires-python = ">= 3.9"
readme = "README.md"

# -----------------------------------------------------------------------------
# Directories, relative to the project root
[paths]
main_dir    = "./"
saving_path = "results"     # where trained models and results are written
data_path   = "data"        # where input datasets live
models      = "models"      # where pretrained augmenters live

# -----------------------------------------------------------------------------
# Input datasets (AnnData .h5ad and gene lists). Keys are referenced by the
# training scripts as `<data_file>_file`, e.g. --data_file Retroseq_updated.
[data]
Dbh_file              = "Dbh_cluster.h5ad"
Retroseq_file         = "retroSeq.h5ad"
Retroseq_updated_file = "retroseq_updated_norm.h5ad"
Dbh_Retroseq_file     = "Dbh_Retroseq_logcpm.h5ad"
hvg_Dbh_file          = "high_var_genes_perbatch.csv"

# -----------------------------------------------------------------------------
# Pretrained VAE-GAN augmenters, keyed by platform (`augmenter_<platform>`).
# Leave empty to train a new augmenter from scratch.
[models]
augmenter_Dbh      = ""
augmenter_Retroseq = ""
```

---

## Usage

Run `python <script>.py --help` to see the full list of hyperparameters for any script.

All scripts are run from the repository root.

**Train the RNA data augmenter (VAE-GAN):**

```bash
python train/train_augmenter.py --cuda --n_epoch 20000
```

**Train the coupled mixture VAE (MMIDAS):**

```bash
python train/train_mmidas.py --n_categories 15 --state_dim 2 --tau 0.1 --cuda
```

**Submit a training job to SLURM:**

```bash
sbatch train/scripts/train_mmidas.sh
```

The `notebooks/` directory contains notebooks for validating the augmenter and inspecting the learned MMIDAS representation.

---

## Repository layout

```text
LC-NE-MixRep/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── train/
│   ├── train_augmenter.py             # train the VAE-GAN data augmenter
│   ├── train_mmidas.py                # train the coupled mixture VAE (MMIDAS)
│   ├── train_aug_mmidas.py            # joint augmenter + MMIDAS training
│   └── scripts/
│       └── train_mmidas.sh            # SLURM submission script
├── notebooks/
│   ├── 1_validate_augmenter.ipynb     # validate the VAE-GAN augmenter
│   └── 2_validate_mmidas.ipynb        # inspect the learned MMIDAS representation
└── mmidas/
    ├── __init__.py
    ├── cplMixVAE.py                   # coupled mixture VAE (core model)
    ├── networks_mixvae.py             # mixture VAE network definitions
    ├── networks_aug.py                # augmenter network definitions
    ├── vaegan.py                      # VAE-GAN augmenter
    ├── eval.py                        # evaluation utilities
    └── utils/
        ├── augmentation.py            # augmentation + data loaders
        ├── batch_removal.py           # batch-effect correction
        ├── cluster_analysis.py        # cluster / consensus analysis
        ├── config_tools.py            # loads the project .toml
        └── data_tools.py              # data loading and dataloaders
```

---

## Data

_Data availability will be added here._

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Citing

_LC-NE paper will be added here._

```bibtex
@article{xxxx,
  title   = {xxx},
  author  = {authors},
  journal = {Nature},
  volume  = {xx},
  number  = {xx},
  pages   = {xxx},
  year    = {xxx},
  publisher = {Nature Publishing Group US New York}
}
```

