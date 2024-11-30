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
# Descriptive Statistic
# provide descriptive based on cleaned data
# option to stack together for macro or '' for detailed

def desc(df, macro):

    if macro: 
        df = df.stack().describe()
        
        return df
    
    else:
        return df.describe()

    # try:
    #     # df = df.mask(df == 0).describe()
    #     df = df.describe()

    #     if macro:
    #         df = df.stack().describe()
            
    #         return df
        
    #     else: return df
    
    # except ValueError as error_msg:
    #     print ('Check your dataframe and type if it''s valid.')
    #     print ('Error message:', error_msg)

def write_desc(df, index, type, macro):    
    if type == 'esg': type = 'ESG Scores'
    elif type == 'e': type = 'E Scores'
    elif type == 's': type = 'S Scores'
    elif type == 'g': type = 'G Scores'
    elif type == 'roe': type = 'Return on Equity (ROE)'
    elif type == 'roa': type = 'Return on Assets (ROA)'
    elif type == 'yoy': type = '52 Week Return'
    elif type == 'q': type = 'Q Ratio'

    filename = index.upper() + ' '

    if macro: filename = filename + type + ' Macro Overview'
    else: filename = filename + type + ' Descriptive Overview'

    try:
        with open (config.results_path + filename + '.csv', 'w') as file:
            file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

        df.to_csv(config.results_path + filename + '.csv',
                    mode = 'a',
                    index=True,
                    header=True)
    
    except FileNotFoundError:
        os.mkdir(config.results_path)

        with open (config.results_path + filename + '.csv', 'w') as file:
            file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

        df.to_csv(config.results_path + filename + '.csv',
                    mode = 'a',
                    index=True,
                    header=True)

def export_file(target, index, macro):
    if target == 'finp': loop_arr = ['roe', 'roa', 'yoy', 'q']
    elif target == 'scores': loop_arr = ['esg', 'e', 's', 'g']
    elif target == 'all': loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'yoy', 'q']
    else: loop_arr = [target]

    if index == 'all':  index_arr = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse']
    else: index_arr = [index]

    for each_index in index_arr:
        for each_measure in loop_arr:
            write_desc(
                df = desc(
                    df = wrangle.output_df(each_index, 
                                        each_measure),
                    macro = macro),
                index = each_index,
                type = each_measure,
                macro = macro
            )


# wrangle.scores(index, measure)
# wrangle.returns(index, measure)
# wrangle.output_df(index, measure)

start = timeit.default_timer()

# export data 
export_file(target='all',
            index='all', 
            macro=True)

export_file(target='all', 
            index='all',
            macro=False)

print ()

stop = timeit.default_timer()

print ('Time taken:', str(stop-start), '\n\n')
