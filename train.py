from helpers.set_up import set_up
from models.unetSpleen import unet_spleen
import sys
from codecarbon import EmissionsTracker

def main():
    logger, project_root = set_up()
    sys.path.append(project_root)
    with EmissionsTracker() as tracker:
        unet_spleen(logger=logger)
    logger.info('Finished running the code')
    
if __name__ == "__main__":
    main() 
