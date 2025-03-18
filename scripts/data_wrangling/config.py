# parameter file

# lib to check platform
import sys, os, logging
from datetime import datetime

# ===========================================================================

# file path - windows
if sys.platform == 'win32':
    data_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/data/"
    script_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/scripts/"
    results_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/"
    desc_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/descriptive/"
    glm_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/"
    
    log = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/logs/"


# file path - mac
elif sys.platform == 'darwin':
    data_path = '/Users/wallace/Documents/code/python/fyp_24/data/'
    script_path = '/Users/wallace/Documents/code/python/fyp_24/scripts/'
    results_path = '/Users/wallace/Documents/code/python/fyp_24/results/'
    desc_path = '/Users/wallace/Documents/code/python/fyp_24/results/descriptive/'
    glm_path = '/Users/wallace/Documents/code/python/fyp_24/results/glm/'

    log = '/Users/wallace/Documents/code/python/fyp_24/logs/'

# ===========================================================================

# File Names
scores = data_path + 'ESG Scores.xlsx'
returns = data_path + 'Financial Returns.xlsx'

# ===========================================================================

# Check if directory exists
path_arr = [results_path, desc_path, glm_path, log]

for path in path_arr:
    os.makedirs(path, exist_ok=True)

# ===========================================================================

# Logging module basic configuration
filename = f"{datetime.today().strftime('%Y%m%d')}_log.txt"
logging.basicConfig(
            filename=log + filename,
            format='[%(asctime)s] [%(levelname)s] = %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode='w',
            level=logging.DEBUG
            )

logging = logging.getLogger(__name__)
logging.info('New session initialized.')
logging.info(f"Log saved in directory with filename {filename}")
logging.info('Logging module successfully initialized.')
logging.info('')
