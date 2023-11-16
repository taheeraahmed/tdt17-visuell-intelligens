import pyfiglet
import logging

import os


def set_up():
    result = pyfiglet.figlet_format("VI babes", font = "slant") 
    print(result) 
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


    LOG_FILE = "logs/log_file.txt"

    logging.basicConfig(level=logging.INFO, 
                format='[%(levelname)s] %(asctime)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ])
    logger = logging.getLogger()
    logger.info('Set-up done')
    return logger, project_root