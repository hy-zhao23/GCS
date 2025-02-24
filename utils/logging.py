import logging
import os
from datetime import datetime
from mpi4py import MPI
import sys

import logging
import os
from datetime import datetime
from mpi4py import MPI
import sys
import logging
import os
from datetime import datetime
from mpi4py import MPI
import sys

def setup_central_logger(name, log_dir='logs', level=logging.DEBUG):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        rank = MPI.COMM_WORLD.Get_rank()
        log_file = os.path.join(log_dir, f"{name}_rank{rank}_{timestamp}.log")
    except:
        rank = 0
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # Create a root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Create file handler and set level to DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler and set level to DEBUG
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - Rank %(rank)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add rank filter to handlers
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
        def filter(self, record):
            record.rank = self.rank
            return True
    
    rank_filter = RankFilter(rank)
    file_handler.addFilter(rank_filter)
    console_handler.addFilter(rank_filter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

# Global logger
logger = setup_central_logger('main')

def get_error_location():
    frame = sys._getframe(2)  # Get the frame of the caller
    return f"Error in {frame.f_code.co_filename}, {frame.f_code.co_name}, line {frame.f_lineno}"

def log_error(message):
    error_location = get_error_location()
    logger.error(f"{message}. {error_location}")

def log_info(message):
    logger.info(message)

def log_warning(message):
    logger.warning(message)

def log_debug(message):
    logger.debug(message)