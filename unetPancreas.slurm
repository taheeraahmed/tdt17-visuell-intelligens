#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --job-name="pancreas_UNETR"
#SBATCH --output=models/Pancreas/UNETR/unet_pancreas_UNETR.out
#SBATCH --gres=gpu:1

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module load Python/3.10.8-GCCcore-12.2.0

pip install -q -r requirements.txt

python train.py -id $SLURM_JOB_ID -m "unet_pancreas" -v "no_augmentaion"

echo "Done running train.py"