# parameter file

# lib to check platform
import sys, os, logging
from datetime import datetime

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

# esg files
nasdaq_esg = data_path + 'ESG_Scores_NASDAQ100_20Y.xlsx'
snp_esg = data_path + 'ESG_Scores_SNP500_20Y.xlsx'
stoxx_esg = data_path + 'ESG_Scores_STOXX600_20Y.xlsx'
msci_esg = data_path + 'ESG_Scores_MSCIWorld_20Y.xlsx'
ftse_esg = data_path + 'ESG_Scores_FTSE350_20Y.xlsx'

# return files 
nasdaq_returns = data_path + 'Returns_NASDAQ100_20Y.xlsx'
snp_returns = data_path + 'Returns_SNP500_20Y.xlsx'
stoxx_returns = data_path + 'Returns_STOXX600_20Y.xlsx'
msci_returns = data_path + 'Returns_MSCIWorld_20Y.xlsx'
ftse_returns = data_path + 'Returns_FTSE350_20Y.xlsx'

# GICS Mapping Table
# MSCI, 2025. The Global Industry Classification Standard (GICS®) 
# Available at: https://www.msci.com/our-solutions/indexes/gics (Accessed: 13 March 2025)

gics = data_path + 'GICS_table.xlsx'
gics_industry_name = 'Information Technology'

# logging basic config
os.makedirs(log, exist_ok=True)

filename = f"{datetime.today().strftime('%Y%m%d')}_log.txt"
logging.basicConfig(
            filename=log + filename,
            format='[%(asctime)s] [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode='a',
            level=logging.DEBUG
            )

logging = logging.getLogger(__name__)
logging.info('New session initialized.')
logging.info(f"Log saved in directory with filename {filename}")
logging.info('Logging module successfully initialized.')
logging.info('')
