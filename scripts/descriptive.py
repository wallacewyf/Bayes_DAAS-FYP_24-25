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

# ------------------------------------------------------------------------
# Descriptive Statistics

def desc(index, measure, macro):    
    """
    Parameters: 
    index = e.g. msci / nasdaq
    measure = e.g. esg / roe / mktcap
    macro = stack or year by year

    Creates descriptive statistics files saved in respective index's directory in .../results/

    It can either create one or multiple files, depending on the argument passed into index / measure variable 

    For all statistics aggregated by index, please run the agg_desc() function below

    """
    mapping = {
        'finp': ['roe', 'roa', 'q', 'ta', 'mktcap'],
        'scores': ['esg', 'e', 's', 'g'],
        'all': ['esg', 'e', 's', 'g', 'roe', 'roa', 'q', 'ta', 'mktcap']
    }

    measure = mapping.get(measure, [measure])
    index = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse'] if index == 'all' else [index]

    for each_index in index:
        for each_measure in measure:
            log.info(f"Writing descriptive stats for {each_index.upper()}'s {each_measure.upper()} data")

            if each_measure in ['esg', 'e', 's', 'g']:
                df = wrangle.scores(each_index, each_measure)
            
            elif each_measure in ['roe', 'roa', 'q', 'ta', 'mktcap']:
                df = wrangle.returns(each_index, each_measure)
            
            else:
                log.warning(f"Unknown measure entered: {each_measure}")
                continue
            
            filename = (f"{each_index.upper()} {each_measure.upper()} {'Macro Overview' if macro else 'Descriptive Overview'}.csv")
            df = df.stack().describe() if macro else df.describe()

            output_dir = f"{config.desc_path}/{each_index.upper()}/"
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, filename), 'w') as file:
                file.write(f"{each_measure.upper()} Descriptive Statistics from 2023 to 2004\n")
            
            df.to_csv(os.path.join(output_dir, filename), mode='a', index=True, header=True)

            log.info(f"{filename} saved in .../results/ directory")
            

# Aggreagetd Descriptive Statistics
def agg_desc(type):
    """
    Parameters: 
    type = e.g. finp / scores/ all

    Creates aggregated by index descriptive statistics file to be saved in .../results/ directory 

    It can either create one or multiple files, depending on the argument passed into "type" variable     
    """
        
    log.info(f'Descriptive Statistics called for {type} scores')

    index_arr = ['msci', 'nasdaq', 'snp', 'stoxx', 'ftse']

    if type == 'scores': loop_arr = ['esg', 'e', 's', 'g']
    elif type == 'finp': loop_arr = ['roe', 'roa', 'mktcap', 'q']
    elif type == 'all': loop_arr = ['esg', 'e', 's', 'g','roe', 'roa', 'mktcap', 'q']
    else: 
        loop_arr = [type]

    for each_measure in loop_arr:
        dfs = []

        for each_index in index_arr:
            log.info(f"Accessing {each_measure.upper()} data from {each_index.upper()}'s dataframe")
            df = wrangle.output(each_index, each_measure).stack().describe()
            
            df.rename(each_index)
            dfs.append(df)
            
        log.info(f"Merging all {each_measure.upper()} data into a single dataframe...")

        df = pd.concat(dfs, axis=1)
        df.columns = ['MSCI World', 
                    'NASDAQ 100',
                    'S&P 500',
                    'STOXX 600',
                    'FTSE 350']
        
        
        ''' 
        Merge and retain only companies that exists throughout
        '''
    

        log.info(f'Exporting {each_measure.upper()} dataframe into csv file in .../descriptive')
        filename = f"{each_measure.upper()} Aggregated Statistics.csv"

        os.makedirs(config.desc_path, exist_ok=True)
        df.to_csv(os.path.join(config.desc_path, filename),
                    header=True,
                    index=True)
            
        log.info (f"{filename} saved to .../results/descriptive directory")
        
        df.drop(df.index, inplace=True)

# codespace starts here
# ----------------------------------------------------------------------

# agg_desc('scores')