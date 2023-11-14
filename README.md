# Direction 

- Goal: Segmentation on two organs and two models using the Decathlon datasets.

Today: 
- Se på notebook-en prøve å få den til å kjøre UNET
    - Se på et annet (2) organ som får det til å kjøre :))
    - Se på et (3) organ
- Se på notebook-en, prøve å få den  til å kjøre med UNET-R
    - (1) Spleen segmentation m/UNET-R


## How to run notebook on IDUN 😮‍💨

(1) Log in to IDUN

````
ssh USERNAME@idun-login1.hpc.ntnu.no
````

(2) Create a new slurm-file 

````
#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=YOUR-ACCOUNT
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --job-name="JOB_NAME"
#SBATCH --output=jupyter.out
#SBATCH --gres=gpu:1
module load Anaconda3/2022.05
jupyter notebook --no-browser
````

(3) Run the slurm-file `sbatch slurm-file.slurm`

````
$ sbatch slurm-file.slurm
Submitted batch job 123456
````

(4) You need the node name, this is how you do it: 
````
$ scontrol show job 123456
JobId=123456 JobName=JOB_NAME
. . .
   JobState=RUNNING Reason=None Dependency=(null)
. . .
   NodeList=NODE_NAME
. . .
   StdOut=/cluster/home/USER/jupyter.out
````

(5) Open new terminal DO NOT CLOSE THIS!

````
ssh -L 88xx:127.0.0.1:88xx -J USER@idun-login1.hpc.ntnu.no USER@NODE-NAME
````

(6) In order to now find the link to open the notebook you need to look at the output file from the slurm-job. 

````
$ cat /cluster/home/USER/jupyter.out
````

(7) At the end of the file, you will find a link -- Paste  this in  your web browser

GOOD JOB 💗
