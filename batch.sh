#!/bin/bash

#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1

cd /home/ptenkaate/scratch/Master-Thesis

echo $$

python3 test_file.py