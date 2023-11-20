# Direction 

## TODOs:
- Goal: Segmentation on two organs and two models using the Decathlon datasets.

Organs to segment: 
1. Spleen on UNET
2. Pancreas on UNET
3. Liver on UNET

Today: 
[ ] Se bildene som lagres, finne ut hvordan vi kan lagre og se de
[X] Se på organ nr.3 
[ ] Finne ut hva vi skal se på: Data augmentations? Andre modeller?   

## Report from Frank 

1. Data analysis!! 
2. Explain stuff you are doing :)) 
3. Discuss the results
4. Make video/explain the results

Fra Frank: Skal mye til å få lavere enn 40%, så lenge vi har vist at vi har prøvd og gjort det vi skal (3 tasks) og forklarer det vi har gjort ordentlig, ikke bare fått det til å kjøre:,)

## Overview of taks

![MSD tasks](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-022-30695-9/MediaObjects/41467_2022_30695_Fig1_HTML.png)

| Task Number | Organ              |
|------------:|:------------------:|
| 01          | BrainTumour        |
| 02          | Heart              |
| 03          | Liver              |
| 04          | Hippocampus        |
| 05          | Prostate           |
| 06          | Lung               |
| 07          | Pancreas           |
| 08          | HepaticVessel      |
| 09          | Spleen             |
| 10          | Colon              |

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

(4) You need the **node name**, this is how you do it: 
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

(5) In order to now find the link to open the notebook you need to look at the output file from the slurm-job. 

````
$ cat /cluster/home/USER/jupyter.out
````

(6) Open new terminal DO NOT CLOSE THIS!

Here you need to add portnumber found in the output file. Also add the NODE-NAME from the `scontrol show job 123456` command.
````
ssh -L 8889:127.0.0.1:8889 -J taheeraa@idun-login1.hpc.ntnu.no taheeraa@idun-04-08
````

(7) At the end of the output file which was opened in step 5 you will find a link -- Paste  this in  your web browser

GOOD JOB 💗
