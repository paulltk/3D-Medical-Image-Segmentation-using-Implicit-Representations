#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1

cd /home/ptenkaate/scratch/Master-Thesis/pi-gan_sequential

python3 seq_pi_gan.py --cnn_setup=-1 --mapping_setup=7 --dataset=full