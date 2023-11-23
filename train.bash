#!/bin/bash

# Generate a unique identifier
UNIQUE_ID=$(date +%Y%m%d-%H%M%S)
AUGMENTATION="rand_affine"
MODEL="unet_spleen"

# Define the destination path for the code
CODE_PATH="/cluster/home/taheeraa/runs/$AUGMENTATION-$MODEL-$UNIQUE_ID"

# Copy the code with rsync, excluding .venv
echo "Copying code to $CODE_PATH"
mkdir -p $CODE_PATH
rsync -av --exclude='.venv' --exclude='logs' --exclude='idun' --exclude='emissions.csv' --exclude='notebooks' --exclude='outputs' /cluster/home/taheeraa/code/tdt17-visuell-intelligens/ $CODE_PATH

# Copy the SLURM script to the same location
cp /path/to/unetSpleen.slurm $CODE_PATH/

# Submit the job to SLURM with the necessary environment variables
echo "Running slurm job from $CODE_PATH"
sbatch --export=UNIQUE_ID=$UNIQUE_ID --export=AUGMENTATION=$AUGMENTATION --export=MODEL=$MODEL --export=CODE_PATH=$CODE_PATH $CODE_PATH/unetSpleen.slurm