# import parameter file
import config

# import libraries
import pandas as pd
import numpy as np
import os
import logging

# initialize logger module 
log = config.logging

col_names = list(range(2023, 2003, -1))

industry_list = [
    'IT Services',
    'Software',
    'Communications Equipment',
    'Technology Hardware, Storage & Peripherals',
    'Electronic Equipment, Instruments & Components',
    'Semiconductors & Semiconductor Equipment'
]

# Notes: 

# dropna not needed - intersection should be ran first and then NA dropped

def returns(index, measure):
    # path = filepath from config
    # measure = measure of table requested

    if index == 'nasdaq': 
        logging.info('Retrieving NASDAQ data...')
        data = config.nasdaq_returns
    elif index == 'snp': 
        logging.info('Retrieving S&P 500 data...')
        data = config.snp_returns
    elif index == 'stoxx': 
        logging.info('Retrieving STOXX data...')
        data = config.stoxx_returns
    elif index == 'ftse': 
        logging.info('Retrieving FTSE data...')
        data = config.ftse_returns
    elif index == 'msci': 
        logging.info('Retrieving MSCI data...')
        data = config.msci_returns
    else: 
        print ('Invalid index!')
        return

    returns_df = pd.read_excel(data, index_col=[0,1,2,3], skiprows=2)
    returns_df.index.names = ['Identifier', 'Company Name', 'GICS Industry Name', 'Exchange Name']
    returns_df.sort_values(by='Company Name',
                            axis=0,
                            ascending=True,
                            inplace=True)

    if measure == 'all':
        logging.info ('Extracting all financial returns...')
        return returns_df
    
    elif measure == 'roe':
        # Return on Equity - Actual
        logging.info ('Extracting Return on Equity...')

        roe_df = returns_df.iloc[:, :20]
        roe_df = roe_df.set_axis(col_names, axis=1)
        roe_df = roe_df.iloc[:, ::-1]

        # extract only Tech components
        roe_df = roe_df.loc[roe_df.index.get_level_values('GICS Industry Name').isin(industry_list)]
        # roe_df.dropna(inplace=True)

        return roe_df
    
    elif measure == 'roa':
        # Return on Assets - Actual
        logging.info ('Extracting Return on Assets...')

        roa_df = returns_df.iloc[:, 20:40]
        roa_df = roa_df.set_axis(col_names, axis=1)
        roa_df = roa_df.iloc[:, ::-1]

        # extract only Tech components
        roa_df = roa_df.loc[roa_df.index.get_level_values('GICS Industry Name').isin(industry_list)]
        # roa_df.dropna(inplace=True)

        return roa_df

    elif measure == 'yoy':
        # 52-Week Total Return
        yoy_return = returns_df.iloc[:, 40:60]
        yoy_return = yoy_return.set_axis(col_names, axis=1)
        yoy_return = yoy_return.iloc[:, ::-1]
        # yoy_return.dropna(inplace=True)

        # extract only Tech components
        yoy_return = yoy_return.loc[yoy_return.index.get_level_values('GICS Industry Name').isin(industry_list)]

        return yoy_return
    
    elif measure == 'mktcap':
        # Market Capitalisation

        logging.info( 'Extracting Market Capitalisation...')

        mkt_cap = returns_df.iloc[:, 60:80]
        mkt_cap = mkt_cap.set_axis(col_names, axis=1)
        mkt_cap = mkt_cap.iloc[:, ::-1]
        # mkt_cap.dropna(inplace=True)

        # extract only Tech components
        mkt_cap = mkt_cap.loc[mkt_cap.index.get_level_values('GICS Industry Name').isin(industry_list)]

        return mkt_cap

    elif measure == 'ta':
        # Total Assets - Reported
        ta_df = returns_df.iloc[:, 80:100]
        ta_df = ta_df.set_axis(col_names, axis=1)
        ta_df = ta_df.iloc[:, ::-1]
        # ta_df.dropna(inplace=True)

        return ta_df

    elif measure == 'q':
        # Q Ratio
        # Total Assets - Reported

        logging.info ('Extracting Q Ratio...')

        ta_df = returns_df.iloc[:, 80:100]
        ta_df = ta_df.set_axis(col_names, axis=1)
        ta_df.dropna(inplace=True)

        # Market Capitalisation
        mkt_cap = returns_df.iloc[:, 60:80]
        mkt_cap = mkt_cap.set_axis(col_names, axis=1)
        mkt_cap.dropna(inplace=True)        
        
        logging.info ('Aligning dataframes for Q Ratio calculation...')

        q_ta, q_mktcap = ta_df.align(mkt_cap, join='inner')
        q_ratio = q_mktcap / q_ta
        q_ratio = q_ratio.iloc[:, ::-1]

        # extract only Tech components
        q_ratio = q_ratio.loc[q_ratio.index.get_level_values('GICS Industry Name').isin(industry_list)]

        logging.info ('Q Ratio calculation completed.')

        return q_ratio
    
def scores(index, measure):
    # path = filepath from config
    # measure = measure of table requested

    if index == 'nasdaq': data = config.nasdaq_esg
    elif index == 'snp': data = config.snp_esg
    elif index == 'stoxx': data = config.stoxx_esg
    elif index == 'ftse': data = config.ftse_esg
    elif index == 'msci': data = config.msci_esg
    else: 
        print ('Invalid index!')
        return

    source_df = pd.read_excel(data, index_col=[0,1,2,3], skiprows=2)
    source_df.index.names = ['Identifier', 
                             'Company Name', 
                             'GICS Industry Name', 
                             'Exchange Name']

    column_names = col_names

    if measure == 'esg':
        log.info(f'Extracting ESG Scores...')

        esg_scores = source_df.iloc[:, :20]
        esg_scores = esg_scores.set_axis(column_names, axis=1)
        # esg_scores.dropna(inplace=True)
        esg_scores.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
        
        esg_scores = esg_scores.iloc[:, ::-1]
        
        esg_scores = esg_scores.loc[esg_scores.index.get_level_values('GICS Industry Name').isin(industry_list)]
        
        return esg_scores
    
    elif measure == 'e':
        log.info('Extracting E scores...')

        e_pillars = source_df.iloc[:, 20:40]
        e_pillars = e_pillars.set_axis(column_names, axis=1)
        # e_pillars.dropna(inplace=True)
        e_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
        
        e_pillars = e_pillars.iloc[:, ::-1]
        
        e_pillars = e_pillars.loc[e_pillars.index.get_level_values('GICS Industry Name').isin(industry_list) ]

        return e_pillars

    elif measure == 's':
        log.info('Extracting S scores...')
        s_pillars = source_df.iloc[:, 40:60] 
        s_pillars = s_pillars.set_axis(column_names, axis=1)
        # s_pillars.dropna(inplace=True)
        s_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)        
        
        s_pillars = s_pillars.iloc[:, ::-1]
        
        s_pillars = s_pillars.loc[s_pillars.index.get_level_values('GICS Industry Name').isin(industry_list)]
                    
        return s_pillars
    
    elif measure == 'g':
        log.info('Extracting G scores...')
        g_pillars = source_df.iloc[:, 60:80]
        g_pillars = g_pillars.set_axis(column_names, axis=1)
        # g_pillars.dropna(inplace=True)
        g_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
    
        g_pillars = g_pillars.iloc[:, ::-1]
        g_pillars = g_pillars.loc[g_pillars.index.get_level_values('GICS Industry Name').isin(industry_list)]

        return g_pillars
    
def output_df(index, measure):
    if measure in ['roe', 'roa', 'q', 'yoy', 'mktcap', 'ta']:
        return returns(index, measure)
    
    elif measure in ['esg', 'e', 's', 'g']:
        return scores(index, measure)
    

def debug(index, measure, type):
    while True:
        try:
            if type == 'csv':
                output_df(index, measure).to_csv(config.results_path + index + '_' + measure + '.csv',
                                                header=True,
                                                index=True)

            elif type == 'excel':
                output_df(index, measure).to_excel(config.results_path + index + '_' + measure + '.xlsx',
                                                header=True,
                                                index=True)
                
        
        except OSError: 
            os.mkdir(config.results_path)

        break

def export(df, filename, type):
    while True: 
        try:
            if type == 'csv':
                log.info(f'Exporting {filename} to CSV...')

                df.to_csv(config.results_path + filename + '.csv',
                          header=True, 
                          index=True)

            elif type == 'excel':
                df.to_excel(config.results_path + filename + '.xlsx',
                          header=True, 
                          index=True)
        
        except OSError: 
            os.mkdir(config.results_path)

            continue

        break