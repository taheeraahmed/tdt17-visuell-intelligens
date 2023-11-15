#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=15G
#SBATCH --job-name="vi"
#SBATCH --output=idun/unetliver-v2.out
#SBATCH --gres=gpu:1

pwd

module load Python/3.10.8-GCCcore-12.2.0

pip install -q -r requirements.txt

python models/unetLiver.py
