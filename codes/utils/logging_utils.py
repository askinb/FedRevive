import logging
import os
from datetime import datetime

def setup_logger(name, log_dir=None, level=logging.INFO, resume=False):
    """
    Setup logger with the given name and level.
    
    Args:
        name (str): Name of the logger
        log_dir (str): Directory to save log files. If None, logs are only printed to stdout.
        level (int): Logging level
        resume (bool): If True, the logger will resume from the last log file.
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Prevent propagation to the root logger
    logger.propagate = False
    
    # Create formatter with simplified time format (no date, no milliseconds)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    #                              datefmt='%H:%M:%S')
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        # Create the log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file path
        log_file = os.path.join(log_dir, "out.log")
        
        # Remove existing log file if it exists and resume is False
        if os.path.exists(log_file):
            if resume:
                pass
            else:
                os.remove(log_file)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 