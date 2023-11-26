#!/bin/bash

# Generate a unique identifier
DATE=$(date +%Y%m%d-%H%M%S)
AUGMENTATION="baseline" # "rand_affine", "rand_noise", "rand_gamma", "baseline"
MODEL="unetr"           # "unet", "unetr"
ORGAN="spleen"          # "spleen", "liver", "pancreas"
USER=$(whoami)
echo "Current user is: $USER"

ID="${AUGMENTATION}-${ORGAN}-${MODEL}-${DATE}"

JOB_NAME=$ID
OUTPUT_FILE="/cluster/home/taheeraa/runs/idun_out/${ID}.out"

# Define the destination path for the code
CODE_PATH="/cluster/home/$USER/runs/code/${ID}"

# Copy the code with rsync, excluding .venv
echo "Copying code to $CODE_PATH"
mkdir -p $CODE_PATH

rsync -av \
  --exclude='.venv' \
  --exclude='logs' \
  --exclude='models' \
  --exclude='idun' \
  --exclude='emissions.csv' \
  --exclude='notebooks' \
  --exclude='outputs' \
  --exclude='seg_models/__pycache__' \
  --exclude='helpers/__pycache__' \
  --exclude='images/' \
  --exclude='.git' \
  /cluster/home/$USER/code/tdt17-visuell-intelligens/ $CODE_PATH

# Submit the job to SLURM with the necessary environment variables
echo "Running slurm job from $CODE_PATH"
sbatch --partition=GPUQ \
  --account=ie-idi \
  --time=00:15:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --mem=50G \
  --gres=gpu:1 \
  --job-name=$JOB_NAME \
  --output=$OUTPUT_FILE \
  --export=DATE=$DATE,AUGMENTATION=$AUGMENTATION,MODEL=$MODEL,CODE_PATH=$CODE_PATH,USER=$USER,ORGAN=$ORGAN \
  $CODE_PATH/train.slurm
