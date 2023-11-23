#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --gres=gpu:1

echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

echo "--- nvidia-smi ---"
nvidia-smi
module load Python/3.10.8-GCCcore-12.2.0
pip install -q -r requirements.txt

echo "--- Running python-script ---"
echo "Running python script with Model: $MODEL and Augmentation: $AUGMENTATION"
echo "Path to code: $CODE_PATH"
python $CODE_PATH/train.py -id $AUGMENTATION-$UNIQUE_ID -m $MODEL -a $AUGMENTATION

echo " --- Done running train.py ---"