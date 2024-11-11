# import parameter file
import config

# import libraries
import eikon as ek
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt

returns_df = pd.read_excel(config.returns, skiprows=4, index_col=0)
returns_df.drop(returns_df.index[1:2], axis=0, inplace=True)
returns_df.drop(returns_df.index[14:], axis=0, inplace=True)
returns_df.drop(returns_df.index[1], axis=0, inplace=True)

# returns on equity
roe_df = returns_df.loc[['ROE Comm Eqty, %, TTM']]
roe_df = roe_df.T
roe_df.rename(columns={'ROE Comm Eqty, %, TTM': 'ROE (%)'}, inplace=True)

# returns on assets
roa_df = returns_df.loc[['ROA Tot Assets, %, TTM']]
roa_df = roa_df.T
roa_df.rename(columns={'ROA Tot Assets, %, TTM': 'ROA (%)'}, inplace=True)

print (roe_df)
print ()
print (roa_df)

ek.set_app_id('34b17369973d4e5eb52450fa0de997bc41a13d13')

df = ek.get_timeseries(["MSFT.O"], 
                       start_date="2016-01-01",  
                       end_date="2016-01-10")
print (df)