#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH -w, --nodelist=rng-flux-01

cd /home/ptenkaate/scratch/Master-Thesis

python3 pi_gan.py --epochs=100 --cnn_setup=21