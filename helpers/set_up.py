import pyfiglet
import logging
import os
from datetime import datetime
from helpers.create_dir import create_directory_if_not_exists

def set_up():
    result = pyfiglet.figlet_format("VI babes", font = "slant") 
    print(result) 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    create_directory_if_not_exists("logs")
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    LOG_FILE = f"logs/{dt_string}.txt"

    logging.basicConfig(level=logging.INFO, 
                format='[%(levelname)s] %(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger()
    logger.info('Set-up done')
    return logger, project_root