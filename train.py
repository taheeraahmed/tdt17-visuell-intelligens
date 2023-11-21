from helpers.set_up import set_up
from models.unetSpleen import unet_spleen
from models.unetLiver import unet_liver
from models.unetPancreas import unet_pancreas
import sys
import argparse
from codecarbon import EmissionsTracker

def main(args):
    logger, project_root = set_up()
    sys.path.append(project_root)
    
    job_id = args.id

    with EmissionsTracker() as tracker:
        if args.model == "unet_spleen":
            logger.info('Running unet_spleen')
            unet_spleen(logger=logger, job_id=job_id)
        elif args.model == "unet_liver":
            logger.info('Running unet_liver')
            unet_liver(logger=logger)
        elif args.model == "unet_pancreas":
            logger.info('Running unet_pancreas')
            unet_pancreas(logger=logger)
        else:
            logger.error("Invalid model selected")
            sys.exit(1)    
    logger.info('Finished running the code')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net  MSD")
    parser.add_argument("-id", "--id", help="Id of job on IDUN", type=int, default=0, required=False)
    parser.add_argument("-m", "--model", choices=["unet_spleen", "unet_liver", "unet_pancreas"], help="Model to run", required=True)
   
    args = parser.parse_args()
    main(args)
