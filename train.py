from helpers.set_up import set_up
from models.unetSpleen import unet_spleen
from codecarbon import track_emissions

import sys

@track_emissions()
def main():
    logger, project_root = set_up()
    sys.path.append(project_root)
    
    unet_spleen(logger=logger)
    logger.info('Finished running the code')
    
if __name__ == "__main__":
    main() 
