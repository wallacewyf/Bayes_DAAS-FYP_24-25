# parameter file

# lib to check platform
import sys, os, logging

# file path - windows
if sys.platform == 'win32':
    data_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/data/"
    script_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/scripts/"
    results_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/"
    log = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/logs/"

# file path - mac
elif sys.platform == 'darwin':
    data_path = '/Users/wallace/Documents/code/python/fyp_24/data/'
    script_path = '/Users/wallace/Documents/code/python/fyp_24/scripts/'
    results_path = '/Users/wallace/Documents/code/python/fyp_24/results/'
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

# logging basic config
try:
    logging.basicConfig(
                        filename=log + "/log.txt",
                        format='[%(asctime)s] [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a',
                        level=logging.DEBUG
                        )

except FileNotFoundError:
    os.mkdir(log)

logging = logging.getLogger(__name__)
logging.info('Logging module initialized.')