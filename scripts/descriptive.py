# packages
import sys, os, datetime
import pandas as pd

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# import packages
import config, wrangle

# initialize logger module 
log = config.logging

# initialize runtime module
start = datetime.datetime.now()

# Notes
# """
# work on cleaning up functions since they're quite redundant
# """

# ------------------------------------------------------------------------
# Descriptive Statistics
# NA's are automatically ignored from pd.describe() function
# (cited from: Pandas Documentation)

# this should be combined with write_desc() and export_desc()
def desc(df, macro):
    if macro: 
        df = df.stack().describe()
        
        return df
    
    else:
        return df.describe()

# should be combined with export_desc() below
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
        with open (config.desc_path + filename + '.csv', 'w') as file:
            file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

        result_df = desc(df, macro)

        log.info (f'{filename} saved in .../results/ directory')

        result_df.to_csv(config.desc_path + filename + '.csv',
                    mode = 'a',
                    index=True,
                    header=True)
    
    except FileNotFoundError:
        os.makedirs(config.desc_path, exist_ok=True)

        with open (config.desc_path + filename + '.csv', 'w') as file:
            file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

        result_df = desc(df, macro)

        log.info (f'{filename} saved in .../results/ directory')
        result_df.to_csv(config.desc_path + filename + '.csv',
                    mode = 'a',
                    index=True,
                    header=True)

# Descriptive Statistics for ONE measure ONLY
# After testing - pretty redundance since macro_desc does the same thing
def export_file(target, index, macro):
    if target == 'finp': loop_arr = ['roe', 'roa', 'yoy', 'q', 'ta' ,'mktcap']
    elif target == 'scores': loop_arr = ['esg', 'e', 's', 'g']
    elif target == 'all': loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'yoy', 'q', 'ta' ,'mktcap']
    else: loop_arr = [target]

    if index == 'all':  index_arr = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse']
    else: index_arr = [index]

    for each_index in index_arr:
        for each_measure in loop_arr:
            log.info(f"Writing descriptive stats for {each_index.upper()}'s {each_measure.upper()} data")

            write_desc(df = wrangle.output_df(each_index, each_measure), 
                       index = each_index,
                        type = each_measure,
                        macro = macro)

export_file(target='finp',
            index='msci', 
            macro=False)

# Descriptive Statistics for all of the years accordingly to measure and index
def macro_desc(type, format):
    log.info(f'Descriptive Statistics called for {type} scores')

    index_arr = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse']

    if type == 'esg': loop_arr = ['esg', 'e', 's', 'g']
    elif type == 'finp': loop_arr = ['roe', 'roa', 'yoy', 'mktcap', 'q']
    elif type == 'all': loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'yoy', 'mktcap', 'q']

    for each_measure in loop_arr:
        dfs = []

        for each_index in index_arr:
            log.info(f"Accessing {each_measure.upper()} data from {each_index.upper()}'s dataframe")
            df = desc(df=wrangle.output_df(each_index, each_measure),
                        macro=True)
            
            df.rename(each_index)
            dfs.append(df)
            
        log.info(f"Merging all {each_measure.upper()} data into a single dataframe...")

        df = pd.concat(dfs, axis=1)
        df.columns = ['MSCI World', 
                    'NASDAQ 100',
                    'S&P 500',
                    'STOXX 600',
                    'FTSE 350']
        
        log.info(f'Exporting {each_measure.upper()} dataframe into csv file in .../results/ path')
        
        os.makedirs(config.desc_path, exist_ok = True)
    
        if format == 'txt':
            output_path = os.path.join(config.desc_path, filename)
            filename = f"{each_measure.upper()}_desc.txt"

            with open (output_path, 'w') as result:
                result.write(df.to_string(header=True, 
                                        index=True))

        elif format == 'csv':
            filename = f"{each_measure.upper()}_desc.csv"

            df.to_csv(config.desc_path + filename,
                      header=True,
                      index=True)
            
        log.info (f"{filename} saved to .../results/desc directory")
        
        df.drop(df.index, inplace=True)

# macro_desc(type='finp', 
#            format='csv')

stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")