# import parameter file
import config

# import libraries
import pandas as pd
import numpy as np

col_names = list(range(2023, 2003, -1))

def returns(path, type):
    # path = filepath from config
    # type = type of table requested

    returns_df = pd.read_excel(path, index_col=[0,1,2,3], skiprows=2)
    returns_df.index.names = ['Identifier', 'Company Name', 'GICS Industry Name', 'Exchange Name']
    returns_df.sort_values(by='Company Name',
                            axis=0,
                            ascending=True,
                            inplace=True)

    if type == 'all':
        return returns_df
    
    elif type == 'roe':
        # Return on Equity  - Actual
        roe_df = returns_df.iloc[:, :20]
        roe_df = roe_df.set_axis(col_names, axis=1)
        roe_df.dropna(inplace=True)
        

        return roe_df
    
    elif type == 'roa':
        # Return on Assets - Actual
        roa_df = returns_df.iloc[:, 20:40]
        roa_df = roa_df.set_axis(col_names, axis=1)
        roa_df.dropna(inplace=True)

        return roa_df

    elif type == 'yoy':
        # 52-Week Total Return
        yoy_return = returns_df.iloc[:, 40:60]
        yoy_return = yoy_return.set_axis(col_names, axis=1)
        yoy_return.dropna(inplace=True)

        return yoy_return
    
    elif type == 'mktcap':
        # Market Capitalisation
        mkt_cap = returns_df.iloc[:, 60:80]
        mkt_cap = mkt_cap.set_axis(col_names, axis=1)
        mkt_cap.dropna(inplace=True)

        return mkt_cap

    elif type == 'ta':
        # Total Assets - Reported
        ta_df = returns_df.iloc[:, 80:100]
        ta_df = ta_df.set_axis(col_names, axis=1)
        ta_df.dropna(inplace=True)

        return ta_df

    elif type == 'tl':
        # Total Liabilities
        tl_df = returns_df.iloc[:, 100:120]
        tl_df = tl_df.set_axis(col_names, axis=1)
        tl_df.dropna(inplace=True)

        return tl_df

    elif type == 'te':
        # Total Equity
        te_df = returns_df.iloc[:, 120:140]
        te_df = te_df.set_axis(col_names, axis=1)
        te_df.dropna(inplace=True)

        return te_df

    elif type == 'mi':
        # Minority Interest
        minority_df = returns_df.iloc[:, 140:160]
        minority_df = minority_df.set_axis(col_names, axis=1)
        minority_df.dropna(inplace=True)

        return minority_df

    elif type == 'pe':
        # P/E Ratio
        pe_df = returns_df.iloc[:, 160:180]
        pe_df = minority_df.set_axis(col_names, axis=1)
        pe_df.dropna(inplace=True)

        return pe_df

    elif type == 'q':
        # Q Ratio
        # Total Assets - Reported
        ta_df = returns_df.iloc[:, 80:100]
        ta_df = ta_df.set_axis(col_names, axis=1)
        ta_df.dropna(inplace=True)

        # Market Capitalisation
        mkt_cap = returns_df.iloc[:, 60:80]
        mkt_cap = mkt_cap.set_axis(col_names, axis=1)
        mkt_cap.dropna(inplace=True)        
        
        q_ta, q_mktcap = ta_df.align(mkt_cap, join='inner')
        q_ratio = q_mktcap / q_ta

        return q_ratio
    
def scores(path, type):
    # path = filepath from config
    # type = type of table requested

    source_df = pd.read_excel(path, index_col=[0,1,2,3], skiprows=2)
    source_df.index.names = ['Identifier', 
                             'Company Name', 
                             'GICS Industry Name', 
                             'Exchange Name']

    column_names = col_names

    if type == 'esg':
        esg_scores = source_df.iloc[:, :20]
        esg_scores = esg_scores.set_axis(column_names, axis=1)
        esg_scores.dropna(inplace=True)
        esg_scores.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
        
        return esg_scores
    
    elif type == 'e':
        e_pillars = source_df.iloc[:, 20:40]
        e_pillars = e_pillars.set_axis(column_names, axis=1)
        e_pillars.dropna(inplace=True)
        e_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)

        return e_pillars

    elif type == 's':
        s_pillars = source_df.iloc[:, 40:60] 
        s_pillars = s_pillars.set_axis(column_names, axis=1)
        s_pillars.dropna(inplace=True)
        s_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)        
                    
        return s_pillars
    
    elif type == 'g':
        g_pillars = source_df.iloc[:, 60:80]
        g_pillars = g_pillars.set_axis(column_names, axis=1)
        g_pillars.dropna(inplace=True)
        g_pillars.sort_values(by=['Company Name', 2023],
                            axis=0,
                            ascending=[True, False],
                            inplace=True)
    
        return g_pillars
    

