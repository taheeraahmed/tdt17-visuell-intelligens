import pyfiglet
import logging
import os
from helpers.create_dir import create_directory_if_not_exists

def set_up(model, unique_id, augmentation):
    result = pyfiglet.figlet_format("VI babes", font = "slant") 
    print(result) 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    LOG_DIR = f"/cluster/home/taheeraa/runs/output/{augmentation}-{unique_id}"
    create_directory_if_not_exists(LOG_DIR)
    LOG_FILE = f"{LOG_DIR}/log.txt"

    logging.basicConfig(level=logging.INFO, 
                format='[%(levelname)s] %(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger()
    logger.info('Set-up done')
    logger.info(f"Logging to file: {LOG_DIR}")
    return logger, project_root