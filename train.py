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
    logger, project_root = set_up()
    sys.path.append(project_root)
    
    job_id = args.id
    model = args.model
    logger.info(f"Job ID: {job_id}")
    
    start_time = time.time()
    
    if check_nvml_error(logger=logger) == 0:
        logger.info('Code carbon is working B)')
        with EmissionsTracker(project_name=model, log_level="error") as tracker:
            run_models(model, logger, job_id=job_id)
    else: 
        logger.warning('Dropped carbon tracker :/')
        run_models(model, logger, job_id=job_id)
        
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logger.info('Finished running the code')

def run_models(model, logger, job_id):
    if model == "unet_spleen":
        logger.info('Running unet_spleen w/random affine')
        unet_spleen(logger=logger, job_id=job_id)
    elif model == "unet_liver":
        logger.info('Running unet_liver')
        unet_liver(logger=logger)
    elif model == "unet_pancreas":
        logger.info('Running unet_pancreas')
        unet_pancreas(logger=logger)
    else:
        logger.error("Invalid model selected")
        sys.exit(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net  MSD")
    parser.add_argument("-id", "--id", help="Id of job on IDUN", type=int, default=0, required=False)
    parser.add_argument("-m", "--model", choices=["unet_spleen", "unet_liver", "unet_pancreas"], help="Model to run", required=True)
   
    args = parser.parse_args()
    main(args)
