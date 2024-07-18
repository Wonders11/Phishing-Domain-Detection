import logging
import os
from datetime import datetime

# format of name of the .log file (which is under log folder) : Date:HH:MM:SS.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# getting current directory path
log_path = os.path.join(os.getcwd(),"logs")

# creating the path
os.makedirs(log_path,exist_ok=True)

LOG_FILEPATH = os.path.join(log_path,LOG_FILE)

logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILEPATH,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)