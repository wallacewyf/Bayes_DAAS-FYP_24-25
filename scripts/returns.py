# import parameter file
import config

# import libraries
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

# returns on assets
roa_df = returns_df.loc[['ROA Tot Assets, %, TTM']]
roa_df = roa_df.T
