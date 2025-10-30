import logging
from utils import PATH_DATA_ROOT

logger = logging.getLogger("processing_logger")
logger.setLevel(logging.DEBUG)
log_file = PATH_DATA_ROOT.joinpath("processing.log")

# Create file handler which logs even debug messages
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
