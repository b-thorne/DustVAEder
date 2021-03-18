#!/bin/bash

module load tensorflow/gpu-2.2.0-py37

srun python dustvaeder/vae.py --mode standard --gin_config ./configs/MHD_ResNet.gin --results_name MHD_ResNet
#srun python dustvaeder/vae.py --mode standard --gin_config ./configs/GNILC_ConvNet.gin --results_name GNILC_ConvNet
#srun python dustvaeder/vae.py --mode standard --gin_config ./configs/MHD_ConvNet.gin --results_name MHD_ConvNet