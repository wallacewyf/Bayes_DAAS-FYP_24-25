# packages
import sys, os, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# import packages
import config, wrangle

# ------------------------------------------------------------------------
# Descriptive Statistics
# provide descriptive based on cleaned data
# option to stack together for macro or '' for detailed

def desc(df, macro):
    if macro: 
        df = df.stack().describe()
        
        return df
    
    else:
        return df.describe()

def write_desc(df, index, type, macro):    
    if type == 'esg': type = 'ESG Scores'
    elif type == 'e': type = 'E Scores'
    elif type == 's': type = 'S Scores'
    elif type == 'g': type = 'G Scores'
    elif type == 'roe': type = 'Return on Equity (ROE)'
    elif type == 'roa': type = 'Return on Assets (ROA)'
    elif type == 'yoy': type = '52 Week Return'
    elif type == 'q': type = 'Q Ratio'
    elif type == 'mktcap': type = 'Market Capitalization'
    elif type == 'ta': type = 'Total Assets'

    filename = index.upper() + ' '

    if macro: filename = filename + type + ' Macro Overview'
    else: filename = filename + type + ' Descriptive Overview'

    try:
        with open (config.results_path + filename + '.csv', 'w') as file:
            file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

        result_df = desc(df, macro)

        result_df.to_csv(config.results_path + filename + '.csv',
                    mode = 'a',
                    index=True,
                    header=True)
    
    except FileNotFoundError:
        os.mkdir(config.results_path)

        with open (config.results_path + filename + '.csv', 'w') as file:
            file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

        result_df = desc(df, macro)

        result_df.to_csv(config.results_path + filename + '.csv',
                    mode = 'a',
                    index=True,
                    header=True)

def export_file(target, index, macro):
    if target == 'finp': loop_arr = ['roe', 'roa', 'yoy', 'q', 'ta' ,'mktcap']
    elif target == 'scores': loop_arr = ['esg', 'e', 's', 'g']
    elif target == 'all': loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'yoy', 'q', 'ta' ,'mktcap']
    else: loop_arr = [target]

    if index == 'all':  index_arr = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse']
    else: index_arr = [index]

    for each_index in index_arr:
        for each_measure in loop_arr:
            write_desc(df = wrangle.output_df(each_index, each_measure), 
                       index = each_index,
                        type = each_measure,
                        macro = macro)

def macro_desc(type):
    index_arr = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse']

    if type == 'esg': loop_arr = ['esg', 'e', 's', 'g']
    elif type == 'finp': loop_arr = ['roe', 'roa', 'yoy', 'q', 'mktcap', 'ta']
    elif type == 'all': loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'yoy', 'q', 'mktcap', 'ta']

    for each_measure in loop_arr:
        dfs = []

        for each_index in index_arr:
            df = desc(df=wrangle.output_df(each_index, each_measure),
                        macro=True)
            
            df.rename(each_index)
            dfs.append(df)
            
        df = pd.concat(dfs, axis=1)
        df.columns = ['MSCI World', 
                    'NASDAQ 100',
                    'S&P 500',
                    'STOXX 600',
                    'FTSE 350']

        print (each_measure)
        print (df)

        df.to_csv(config.results_path + ' ' + each_measure.upper() + ' Descriptive Overview.csv', 
                header=True, 
                index=True)
        
        print ('-------------------------------------------------------------------------------------')

        df.drop(df.index, inplace=True)

# wrangle.scores(index, measure)
# wrangle.returns(index, measure)
# wrangle.output_df(index, measure)
# wrangle.debug(index, measure, filename, type = "csv/excel")

start = timeit.default_timer()

macro_desc('finp')

wrangle.debug('snp', 'ta', 'csv')
wrangle.debug('nasdaq', 'ta', 'csv')

# TODO: error with finp returns
# why is microsoft q ratio at 14672??
# issue with TA col
# TA different between NASDAQ & SNP FOR MICROSOFT
# recommended to extract data again for both

print ()

stop = timeit.default_timer()

print ('Time taken:', str(stop-start), '\n\n')