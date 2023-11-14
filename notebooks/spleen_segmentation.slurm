#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --job-name="spleen segmentation"
#SBATCH --output=idun/spleen-segmentation.out
#SBATCH --gres=gpu:1

module load Anaconda3/2022.05
pip install -r ../requirements.txt

jupyter notebook --no-browser