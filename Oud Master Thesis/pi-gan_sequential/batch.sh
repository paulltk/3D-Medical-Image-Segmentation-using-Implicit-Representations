#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=mps:8
#SBATCH --reservation=gpu

export CUDA_VISIBLE_DEVICES=0

cd /home/ptenkaate/scratch/Master-Thesis/pi-gan_sequential

python3 -u seq_pi_gan.py --cnn_setup=-1 --mapping_setup=-1 --pcmra_epochs=0  --first_omega_0=10.

python3 -u seq_pi_gan.py --cnn_setup=-1 --mapping_setup=-1 --pcmra_epochs=0  --first_omega_0=30.

python3 -u seq_pi_gan.py --cnn_setup=-1 --mapping_setup=-1 --pcmra_epochs=0  --first_omega_0=100.

