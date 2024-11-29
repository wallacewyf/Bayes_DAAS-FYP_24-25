# packages
import sys, os, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

# ------------------------------------------------------------------------
# Descriptive Statistic
# provide descriptive based on cleaned data
# option to stack together for macro or '' for detailed

def desc(df, macro):
    try:
        df = df.mask(df == 0).describe().iloc[1:, ]

        if macro:
            df = df.stack().describe()
            df = df.iloc[1:]

            return df
        
        else: return df
    
    except ValueError as error_msg:
        print ('Check your dataframe and type if it''s valid.')
        print ('Error message:', error_msg)

def desc_output(df, index, type, macro):    
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

    with open (config.results_path + filename + '.csv', 'w') as file:
        file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

    df.to_csv(config.results_path + filename + '.csv',
                mode = 'a',
                index=True,
                header=True)

def export(target, macro_flag):
    if target == 'finp': loop_arr = ['roe', 'roa', 'yoy', 'q']
    elif target == 'esg': loop_arr = ['esg', 'e', 's', 'g']
    else: loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'yoy', 'q']

    index_arr = ['msci', 
                'nasdaq', 
                'snp', 
                'stoxx', 
                'ftse']

    for each_index in index_arr:
        for each_measure in loop_arr:
            desc_output(
                df = desc(
                    df = wrangle.output_df(each_index, 
                                        each_measure),
                    macro = macro_flag),
                index = each_index,
                type = each_measure,
                macro = macro_flag
            )

start = timeit.default_timer()

export(target='all', 
       macro_flag=True)

export(target='all', 
       macro_flag=False)

stop = timeit.default_timer()

print ('Time taken:', str(stop-start), '\n\n')
