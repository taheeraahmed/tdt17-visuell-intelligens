from helpers.set_up import set_up
from helpers.check_nvml_error import check_nvml_error
from models.unetSpleen import unet_spleen
from models.unetLiver import unet_liver
from models.unetPancreas import unet_pancreas
import sys
import argparse
from codecarbon import EmissionsTracker
import time 

def main(args):    
    unique_id = args.id
    model = args.model
    augmentation = args.augmentation

    logger, project_root = set_up(model=model, unique_id=unique_id, augmentations=augmentation)
    sys.path.append(project_root)

    logger.info(f"Unique id: {unique_id}")
    logger.info(f'Running {model} w/augmentations')

    start_time = time.time()
    
    if check_nvml_error(logger=logger) == 0:
        logger.info('Code carbon is working B)')
        with EmissionsTracker(project_name=model, log_level="error") as tracker:
            run_models(model, logger, unique_id=unique_id, augmentation=augmentation)
    else: 
        logger.warning('Dropped carbon tracker :/')
        run_models(model, logger, unique_id=unique_id, augmentation=augmentation)
        
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logger.info('Finished running the code')

def run_models(model, logger, unique_id, augmentation):
    logger.info(f'Running {model} w/{augmentation}')

    if model == "unet_spleen":
        unet_spleen(logger=logger, unique_id=unique_id, augmentation=augmentation, model_arg=model)
    elif model == "unet_liver":
        unet_liver(logger=logger, unique_id=unique_id, augmentation=augmentation, model_arg=model)
    elif model == "unet_pancreas":
        unet_pancreas(logger=logger, unique_id=unique_id, augmentation=augmentation, model_arg=model)
    else:
        logger.error("Invalid model selected")
        sys.exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net  MSD")
    parser.add_argument("-id", "--id", help="Unique-ID from train.bash", default=0, required=False)
    parser.add_argument("-m", "--model", choices=["unet_spleen", "unet_liver", "unet_pancreas", "unetr_spleen"], help="Model to run", required=True)
    parser.add_argument("-a", "--augmentation", choices=["rand_affine", "rand_noise", "rand_gamma", "baseline"], help="Data augmentations", required=False, default="baseline")
   
    args = parser.parse_args()
    main(args)
