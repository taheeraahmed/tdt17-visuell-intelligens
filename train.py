from helpers.set_up import set_up
from models.unetSpleen import unet_spleen
import sys

def main():
    logger, project_root = set_up()
    sys.path.append(project_root)

    unet_spleen(logger=logger)
    
if __name__ == "__main__":
    main()  # Call the main function if this script is executed as the main program
