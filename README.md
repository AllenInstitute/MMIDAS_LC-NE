# LC-NE-MixRep
Learning mixture-based representations of LC-NE neurons, using Mixture Model Inference with Discrete-coupled AutoencoderS (MMIDAS).

# MultiOmix
MultiOmix is a repository for joint mixture modeling of RNA-seq and ATAC-seq data, enabling integrated analysis of transcriptomic and chromatin accessibility profiles in single-cell studies.

## Requirements
- Python >= 3.9
- PyTorch >= 2.4
- CUDA enabled computing device

## Conda environment
Creat a new Conda environment and activate it.
```
conda create -n mixrep python=3.9
conda activate mixrep
```
Clone the repository and change your working directory to the newly cloned repository's path. Then install all required packages listed in the ```requirement.txt``` file.
```
cd <directory you want to place the repo>
git clone https://github.com/AllenInstitute/LC-NE-MixRep
cd LC-NE-MixRep
python -m pip install -r requirements.txt
```
#### Pytorch - CPU only
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
#### Pytorch - GPU
Generally, PyTorch is compatible with NVIDIA GPUs that support CUDA, as CUDA provides the GPU acceleration process. There has been progress in making PyTorch compatible with Apple's chips, but for now the focus is on CUDA-supported NVIDIA GPUs.

Use ```nvidia-smi``` command to get the installed CUDA version.

Select the appropriate PyTorch installation from [here](https://pytorch.org/get-started/previous-versions/), matching your CUDA version.

```
# CUDA 12.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
The final step is to install the package in editable mode by running the following command:
```
python -m pip install -e .
```

## Docker image
You can use a Docker image instead of the Conda environment.
You need --> Ubuntu + PyTorch + CUDA (optional)

#### Create a Dockerfile
Use a test editor like nano, vim, etc. to creat the ```Dockerfile```.
```
vim Dockerfile
```
In the editor, add the following lines:
```
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

# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

## Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run Python
CMD ["python"]
```
Save the file and check that file is saved correctly.
```
cat Dockerfile
```

#### Build the Docker image
Run the following command to build the Docker image named ```mixrep```.
```
docker build -t mixrep .
```

#### Run the Docker contrainer
To run the container interactively:
```
docker run -it mixrep
```
To run a specific Python script (e.g., main.py):
```
docker run -v $(pwd):/app mixrep python main.py
```


