# DustVAEder

Models and datasets for generative foreground modeling. 

## Instructions

You can see the requirements in the [`requirements.txt`](./requirements.txt) file. 

Once you have installed the necessary packages, run files in `dustvaeder/` individually. For example, to preprocess the required datasets, run `python dustvaeder/datasets.py`. In order to run analysis, run `python dustvaeder.py`. Each of these scripts is configurable, see the source for details. (N.B, if you would like to train any of the models here, a GPU will be necessary.)

## On NERSC

In order to run on NERSC, if you have access to the GPU queue, you can run with the available `tensorflow` installation:

```bash
module load cgpu
salloc -C gpu -t 60 -c 10 -G 1 -q interactive -A m1759
module load tensorflow/gpu-2.2.0-py37
srun python dustvaeder/vae.py
```

# Datasets

There are two available datasets so far.

* Planck GNILC 545 GHz. The 545 GHz GNILC map is processed as described in [arXiv:2101.11181](https://arxiv.org/abs/2101.11181). 
* MHD simulations: . 

# Models

## Encoder Models

* *Convolutional Autoencoder*: this model is described in the paper [arXiv:2101.11181](https://arxiv.org/abs/2101.11181).
* *VAE with ResNet encoder*: this is a similar model using a ResNet model for the encoder.

## Decoder Models

## Normalizing Flows

* *IAF*: this is an implementation of the inverse autoregressive flow model described in [arXiv:1502/03509](https://arxiv.org/abs/1502.03509).
