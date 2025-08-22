import logging
import os

def setup_logger(log_dir, logger_name="train_log", log_level=logging.INFO):

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"))
    file_handler.setLevel(log_level)


    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)


    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    return logger, console_handler

def enable_console_logging(logger, console_handler, enable=True):

    if enable:
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler) 
    else:
        if console_handler in logger.handlers:
            logger.removeHandler(console_handler) 
