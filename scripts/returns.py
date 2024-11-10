# import parameter file
import config

# import libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt

returns_df = pd.read_excel(config.returns, index_col=0)

# returns on equity
roe_df = returns_df.loc[['ROE Comm Eqty, %, TTM']]
roe_df = roe_df.T
# roe_df['ROE Comm Eqty, %, TTM'] = roe_df['ROE Comm Eqty, %, TTM'].apply(lambda x: x*100)

# returns on assets
roa_df = returns_df.loc[['ROA Tot Assets, %, TTM']]
roa_df = roa_df.T
# roe_df['ROA Tot Assets, %, TTM'] = roe_df['ROA Tot Assets, %, TTM'].apply(lambda x: x*100)
