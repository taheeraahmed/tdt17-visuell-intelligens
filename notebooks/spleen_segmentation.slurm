#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --job-name="spleen segmentation"
#SBATCH --output=idun/data_exploration.out
#SBATCH --gres=gpu:1

module load Anaconda3/2022.05
pip install -r ../requirements.txt

jupyter notebook --no-browser