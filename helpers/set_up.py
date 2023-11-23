import pyfiglet
import logging
import os
from datetime import datetime
from helpers.create_dir import create_directory_if_not_exists

def set_up(model, unique_id, augmentations):
    result = pyfiglet.figlet_format("VI babes", font = "slant") 
    print(result) 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    LOG_DIR = "/cluster/home/taheeraa/runs/logs"
    create_directory_if_not_exists(LOG_DIR)
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    LOG_FILE = f"{LOG_DIR}/{augmentations}-{model}-{unique_id}.txt"

    logging.basicConfig(level=logging.INFO, 
                format='[%(levelname)s] %(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger()
    logger.info('Set-up done')
    return logger, project_root