from helpers.set_up import set_up
from helpers.check_nvml_error import check_nvml_error
from seg_models.segmentation_model import segmentation_model
import sys
import argparse
from codecarbon import EmissionsTracker
import time 

ORGAN_TASK = {
    'spleen': 'Task09_Spleen',
    'liver': 'Task03_Liver',
    'pancreas': 'Task07_Pancreas'
}

def main(args):    
    unique_id = args.id
    model = args.model
    augmentation = args.augmentation
    user = args.user
    organ_task = ORGAN_TASK[args.organ]
    organ = args.organ

    logger, project_root = set_up(model=model, unique_id=unique_id, augmentations=augmentation)
    sys.path.append(project_root)

    logger.info(f"Unique id: {unique_id}")
    logger.info(f'Running {model} w/augmentations {augmentation} on {organ}')

    start_time = time.time()
    
    if check_nvml_error(logger=logger) == 0:
        logger.info('Code carbon is working B)')
        with EmissionsTracker(project_name=f"{model}-{augmentation}", log_level="error") as tracker:
            segmentation_model(model_arg=model, logger=logger, task=organ_task, unique_id=unique_id, augmentation=augmentation, user=user)
    else: 
        logger.warning('Dropped carbon tracker :/')
        segmentation_model(model_arg=model, logger=logger, task=organ_task, unique_id=unique_id, augmentation=augmentation, user=user)
        
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logger.info('Finished running the code')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net  MSD")
    parser.add_argument("-id", "--id", help="Unique-ID from train.bash", default=0, required=False)
    parser.add_argument("-u", "--user", help="User runnning train.py", required=True)
    parser.add_argument("-m", "--model", choices=["unet", "unetr"], help="Model to run", required=True)
    parser.add_argument("-o", "--organ", choices=["spleen", "liver", "pancreas"], help="Organ to do segmentation on", required=True)
    parser.add_argument("-a", "--augmentation", choices=["rand_affine", "rand_noise", "rand_gamma", "baseline"], help="Data augmentations", required=False, default="baseline")
   
    args = parser.parse_args()
    main(args)
