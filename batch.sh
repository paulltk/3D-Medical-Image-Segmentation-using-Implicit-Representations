#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH -w, --nodelist=rng-flux-01
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

cd /home/ptenkaate/scratch/Master-Thesis

python3 pi_gan.py --epochs=500 --cnn_lr=2e-5 --siren_lr=2e-5