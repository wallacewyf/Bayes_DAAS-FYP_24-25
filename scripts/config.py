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

# lseg source file
source_file = 'FY2023_ESG_Returns.xlsx'
source = data_path + source_file

return_file = 'AAPL_returns.xlsx'
returns = data_path + return_file

# combined ESG scores - lseg.py
combined_esg_filename = 'sorted_combined_esg.csv'
