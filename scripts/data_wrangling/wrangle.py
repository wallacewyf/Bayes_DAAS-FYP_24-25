# import parameter file
import config

# import libraries
import pandas as pd
import numpy as np
import os

# initialize logger module 
log = config.logging

col_names = list(range(2023, 2003, -1))

finance = []
tech = []

# extracting industry list
log.info (f"Extracting GICS Industry Names...")
industry = pd.read_excel(config.gics, skiprows=3, usecols=[1,3,5,7])
industry.ffill(inplace=True)
industry = industry.set_axis(['Sector', 'Industry Group', 'Industry', 'Sub-Industry'],
                            axis=1)
industry.set_index(['Sector'], inplace=True)

log.info (f"GICS Categories {config.gics_industry_name} recognized.")
industry = industry.loc[config.gics_industry_name]
industry_list = industry.iloc[:, 1].unique()

industry_list = np.array([item.replace('(New Name)', '')
                            .replace('(Discontinued)', '')
                            .replace('\n', ' ').strip() for item in industry_list])

log.info (f"Categorizing into {config.gics_industry_name} clusters...")
for sector in config.gics_industry_name:
    if sector == 'Information Technology': 
        tech.extend([item.replace('(New Name)', '')
                              .replace('(Discontinued)', '')
                              .replace('\n', ' ').strip() 
                              for item in industry.loc[sector, 'Industry'].unique()])
    
    elif sector == 'Financials': 
        finance.extend([item.replace('(New Name)', '')
                              .replace('(Discontinued)', '')
                              .replace('\n', ' ').strip() 
                              for item in industry.loc[sector, 'Industry'].unique()])

log.info (f"Industry scope: {industry_list}")
log.info ('')

log.info (f"Finance Cluster: {finance}")
log.info ('')

log.info (f"Tech Cluster: {tech}")
log.info ('')

def returns(index, measure):
    # path = filepath from config
    # measure = measure of table requested

    index, measure = index.lower(), measure.lower()

    if index == 'nasdaq': data = config.nasdaq_returns
    elif index == 'snp': data = config.snp_returns
    elif index == 'stoxx': data = config.stoxx_returns
    elif index == 'ftse': data = config.ftse_returns
    elif index == 'msci': data = config.msci_returns
    else: 
        print ('Invalid index!')
        return

    returns_df = pd.read_excel(data, index_col=[0,1,2,3], skiprows=2)
    # set index names
    returns_df.index.names = ['Identifier', 
                              'Company Name', 
                              'GICS Industry Name', 
                              'Exchange Name']
    
    # filter on specified industry
    log.info (f'Filtering returns_df according to specified industry...')
    returns_df = returns_df.loc[returns_df.index.get_level_values('GICS Industry Name').isin(industry_list)]
    
    # sort Company Name in ascending order
    returns_df.sort_values(by='Company Name',
                            axis=0,
                            ascending=True,
                            inplace=True)
    

    if measure == 'all':
        log.info (f'Extracting all financial returns for {index.upper()}...')
        return returns_df
    
    elif measure == 'roe':
        # Return on Equity - Actual
        log.info (f'Extracting Return on Equity for {index.upper()}...')

        roe_df = returns_df.iloc[:, :20]
        roe_df = roe_df.set_axis(col_names, axis=1)
        roe_df = roe_df.iloc[:, ::-1]

        return roe_df
    
    elif measure == 'roa':
        # Return on Assets - Actual
        log.info (f'Extracting Return on Assets for {index.upper()}...')

        roa_df = returns_df.iloc[:, 20:40]
        roa_df = roa_df.set_axis(col_names, axis=1)
        roa_df = roa_df.iloc[:, ::-1]

        return roa_df

    elif measure == 'mktcap':
        # Market Capitalisation
        log.info(f'Extracting Market Capitalisation for {index.upper()}...')

        mkt_cap = returns_df.iloc[:, 40:60]
        mkt_cap = mkt_cap.set_axis(col_names, axis=1)
        mkt_cap = mkt_cap.iloc[:, ::-1]

        return mkt_cap

    elif measure == 'ta':
        # Total Assets - Reported
        ta_df = returns_df.iloc[:, 60:80]
        ta_df = ta_df.set_axis(col_names, axis=1)
        ta_df = ta_df.iloc[:, ::-1]

        return ta_df

    elif measure == 'q':
        # Q Ratio
        # Total Assets - Reported

        log.info (f'Extracting Q Ratio for {index.upper()}...')

        ta_df = returns_df.iloc[:, 60:80]
        ta_df = ta_df.set_axis(col_names, axis=1)

        # Market Capitalisation
        mkt_cap = returns_df.iloc[:, 40:60]
        mkt_cap = mkt_cap.set_axis(col_names, axis=1)
        
        log.info ('Aligning dataframes for Q Ratio calculation...')

        q_ta, q_mktcap = ta_df.align(mkt_cap, join='inner')
        q_ratio = q_mktcap / q_ta
        q_ratio = q_ratio.iloc[:, ::-1]

        log.info ('Q Ratio calculation completed.')

        return q_ratio
    
def scores(index, measure):
    # path = filepath from config
    # measure = measure of table requested 
    index, measure = index.lower(), measure.lower()

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

    # filter on specified industry
    log.info (f'Filtering source_df according to specified industry...')
    source_df = source_df.loc[source_df.index.get_level_values('GICS Industry Name').isin(industry_list)]

    column_names = col_names

    if measure == 'esg':
        log.info(f'Extracting ESG Scores for {index.upper()}...')

        esg_scores = source_df.iloc[:, :20]
        esg_scores = esg_scores.set_axis(column_names, axis=1)
        esg_scores.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
        
        esg_scores = esg_scores.iloc[:, ::-1]
                
        return esg_scores
    
    elif measure == 'e':
        log.info(f'Extracting E scores for {index.upper()}...')

        e_pillars = source_df.iloc[:, 20:40]
        e_pillars = e_pillars.set_axis(column_names, axis=1)
        e_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
        
        e_pillars = e_pillars.iloc[:, ::-1]
        
        return e_pillars

    elif measure == 's':
        log.info(f'Extracting S scores for {index.upper()}...')
        s_pillars = source_df.iloc[:, 40:60] 
        s_pillars = s_pillars.set_axis(column_names, axis=1)
        s_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)        
        
        s_pillars = s_pillars.iloc[:, ::-1]
                            
        return s_pillars
    
    elif measure == 'g':
        log.info(f'Extracting G scores for {index.upper()}...')
        g_pillars = source_df.iloc[:, 60:80]
        g_pillars = g_pillars.set_axis(column_names, axis=1)
        # g_pillars.dropna(inplace=True)
        g_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
    
        g_pillars = g_pillars.iloc[:, ::-1]

        return g_pillars
    
def output(index, measure):
    if measure in ['roe', 'roa', 'q', 'yoy', 'mktcap', 'ta']:
        return returns(index, measure)
    
    elif measure in ['esg', 'e', 's', 'g']:
        return scores(index, measure)