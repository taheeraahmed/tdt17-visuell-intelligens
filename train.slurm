#!/bin/sh
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
echo "Running python script with Model: $MODEL, augmentation: $AUGMENTATION" and organ: $ORGAN
echo "Path to code: $CODE_PATH"
python $CODE_PATH/train.py -id $AUGMENTATION-$DATE -m $MODEL -a $AUGMENTATION -u $USER -o $ORGAN

echo " --- Done running train.py ---"