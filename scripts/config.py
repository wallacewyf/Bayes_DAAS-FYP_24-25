# parameter file

# lib to check platform
import sys

# file path - windows
if sys.platform == 'win32':
    data_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/raw_data/"
    script_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/scripts/"
    results_path = r"C:/Users/walla/Documents/Sandbox/code/fyp_24/results/"

# file path - mac
elif sys.platform == 'darwin':
    data_path = '/Users/wallace/Documents/code/python/fyp_24/raw_data/'
    script_path = '/Users/wallace/Documents/code/python/fyp_24/scripts/'
    results_path = '/Users/wallace/Documents/code/python/fyp_24/results/'

# example files
example_source = 'FY2023_ESG_Returns.xlsx'
example_source = data_path + example_source
example_return = 'AAPL_returns.xlsx'
example_return = data_path + example_return

# esg files
nasdaq_esg = data_path + 'ESG_Scores_NASDAQ100_20Y.xlsx'
snp_esg = data_path + 'ESG_Scores_SNP500_20Y.xlsx'

# return files 
nasdaq_returns = data_path + 'Returns_NASDAQ100_20Y.xlsx'
snp_returns = data_path + 'Returns_SNP500_20Y.xlsx'

# combined ESG scores - lseg.py
combined_esg_filename = 'sorted_combined_esg.csv'
