#!/bin/bash

# Generate a unique identifier
UNIQUE_ID=$(date +%Y%m%d-%H%M%S)
AUGMENTATION="baseline" # "rand_affine", "rand_noise", "rand_gamma", "baseline"
MODEL="unetr_spleen"       # "unet_spleen", "unet_liver", "unet_pancreas"

JOB_NAME="${AUGMENTATION}-${MODEL}-${UNIQUE_ID}"
OUTPUT_FILE="/cluster/home/taheeraa/runs/idun_out/${AUGMENTATION}-${MODEL}-${UNIQUE_ID}.out"

# Define the destination path for the code
CODE_PATH="/cluster/home/taheeraa/runs/code/$AUGMENTATION-$MODEL-$UNIQUE_ID"

# Copy the code with rsync, excluding .venv
echo "Copying code to $CODE_PATH"
mkdir -p $CODE_PATH
rsync -av --exclude='.venv' --exclude='logs' --exclude='idun' --exclude='emissions.csv' --exclude='notebooks' --exclude='outputs' --exclude='models/__pycache__' --exclude='helpers/__pycache__' --exclude='.git' /cluster/home/taheeraa/code/tdt17-visuell-intelligens/ $CODE_PATH

# Submit the job to SLURM with the necessary environment variables
echo "Running slurm job from $CODE_PATH"
sbatch --partition=GPUQ --account=ie-idi --time=20:00:00 --nodes=1 --ntasks-per-node=1 --mem=50G --gres=gpu:1 --job-name=$JOB_NAME --output=$OUTPUT_FILE --export=UNIQUE_ID=$UNIQUE_ID,AUGMENTATION=$AUGMENTATION,MODEL=$MODEL,CODE_PATH=$CODE_PATH $CODE_PATH/unetSpleen.slurm