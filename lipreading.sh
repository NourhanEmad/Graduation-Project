#!/bin/sh
#SBATCH --job-name=gpu_devices
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=50:00:00
#SBATCH --account=g.shams020

# list-gpu-devices/list.sh (Slurm submission script)

python read_dataset.py
