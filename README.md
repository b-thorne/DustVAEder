# DustVAEder

Models and datasets for generative foreground modeling. 

## Instructions

The suggested way to run this code is through Docker, see below for instructions. 

The code is designed to run files in `src/` individually, rather than being imported. For example, to preprocess the required datasets, run `python src/datasets.py`.  Each of these scripts is configurable, see the source for details. (N.B, if you would like to train any of the models here, a GPU will be necessary.)

## On NERSC

In order to run on NERSC, if you have access to the GPU queue, you can run with the available `tensorflow` installation:

```bash
module load cgpu
salloc -C gpu -t 60 -c 10 -G 1 -q interactive -A m1759
srun shifter --image=bthorne93/dustvaeder /bin/bash
python src/vae.py --mode standard 
```

## Docker / Shifter

We provide a Dockerfile for the environment used in this analysis. This can be built from `./Dockerfile`, or pulled from DockerHub at `bthorne93/dustvaeder`. 

## Jupyter kernel

The file `kernel.json` provided is also useful for using this image in jupyter kenels. On NERSC copy it to:

```
~/.local/share/jupyter/kernels/<my-shifter-kernel>/kernel.json
```

# Datasets

* Planck GNILC 545 GHz. The 545 GHz GNILC map is processed as described in [arXiv:2101.11181](https://arxiv.org/abs/2101.11181). 

# Models

* *Convolutional Autoencoder*: this model is described in the paper [arXiv:2101.11181](https://arxiv.org/abs/2101.11181).
* *VAE with ResNet encoder*: this is a similar model using a ResNet model for the encoder.

## Normalizing Flows

* *IAF*: this is an implementation of the inverse autoregressive flow model described in [arXiv:1502/03509](https://arxiv.org/abs/1502.03509).

