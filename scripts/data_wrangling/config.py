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
    
    # Linear Regression
    basic_lm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/basic_lm/"
    lagged_lm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/lagged_lm/"
    log_lag = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/log_lag/"
    log_lm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/log_lm/"

    threshold_basic_lm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/threshold_basic_lm/"
    threshold_lagged_lm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/threshold_lagged_lm/"
    threshold_log_lag = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/threshold_log_lag/"
    threshold_log_lm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/lm/threshold_log_lm/"

    # GLMs    
    basic_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/basic_glm/"
    lagged_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/lagged_glm/"
    log_lag_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/log_lag_glm/"
    log_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/log_glm/"

    threshold_basic_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/threshold_basic_glm/"
    threshold_lagged_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/threshold_lagged_glm/"
    threshold_log_lag_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/threshold_log_lag_glm/"
    threshold_log_glm = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/glm/threshold_log_glm/"
    
    log = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/logs/"


# file path - mac
elif sys.platform == 'darwin':
    data_path = '/Users/wallace/Documents/code/python/fyp_24/data/'
    script_path = '/Users/wallace/Documents/code/python/fyp_24/scripts/'
    results_path = '/Users/wallace/Documents/code/python/fyp_24/results/'
    desc_path = '/Users/wallace/Documents/code/python/fyp_24/results/descriptive/'

    # Linear Regression
    basic_lm = '/Users/wallace/Documents/code/python/fyp_24/results/lm/basic_lm/'
    lagged_lm = '/Users/wallace/Documents/code/python/fyp_24/results/lm/lagged_lm/'
    log_lag = '/Users/wallace/Documents/code/python/fyp_24/results/lm/log_lag/'
    log_lm = '/Users/wallace/Documents/code/python/fyp_24/results/lm/log_lm/'

    threshold_basic_lm = "/Users/wallace/Documents/code/python/fyp_24/results/lm/threshold_basic_lm/"
    threshold_lagged_lm = "/Users/wallace/Documents/code/python/fyp_24/results/lm/threshold_lagged_lm/"
    threshold_log_lag = "/Users/wallace/Documents/code/python/fyp_24/results/lm/threshold_log_lag/"
    threshold_log_lm = "/Users/wallace/Documents/code/python/fyp_24/results/lm/threshold_log_lm/"

    # GLMs    
    basic_glm = '/Users/wallace/Documents/code/python/fyp_24/results/glm/basic_glm/'
    lagged_glm = '/Users/wallace/Documents/code/python/fyp_24/results/glm/lagged_glm/'
    log_lag_glm = '/Users/wallace/Documents/code/python/fyp_24/results/glm/log_lag_glm/'
    log_glm = '/Users/wallace/Documents/code/python/fyp_24/results/glm/log_glm/'

    threshold_basic_glm = "/Users/wallace/Documents/code/python/fyp_24/results/glm/threshold_basic_glm/"
    threshold_lagged_glm = "/Users/wallace/Documents/code/python/fyp_24/results/glm/threshold_lagged_glm/"
    threshold_log_lag_glm = "/Users/wallace/Documents/code/python/fyp_24/results/glm/threshold_log_lag_glm/"
    threshold_log_glm = "/Users/wallace/Documents/code/python/fyp_24/results/glm/threshold_log_glm/"

    log = '/Users/wallace/Documents/code/python/fyp_24/logs/'

# ===========================================================================

# File Names
scores = data_path + 'ESG Scores.xlsx'
returns = data_path + 'Financial Returns.xlsx'

# ===========================================================================

# Check if directory exists
path_arr = [results_path, desc_path, 
            basic_lm, lagged_lm, log_lm, log_lag, 
            threshold_basic_lm, threshold_lagged_lm, 
            threshold_log_lag, threshold_log_lm,
            threshold_basic_glm, threshold_lagged_glm,
            threshold_log_lag_glm, threshold_log_glm,
            basic_glm, lagged_glm,
            log_lag_glm, log_glm,
            log]

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
