# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score                        # R-squared
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy import stats

import statsmodels.api as sm

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

filepath = config.results_path + 'sandbox/'
index = 'ftse'
filename = f"{index.upper()} Financial Returns.xlsx"

os.makedirs(filepath,
            exist_ok=True)

# """

#     elif type == 'roe': type = 'Return on Equity (ROE)'
#     elif type == 'roa': type = 'Return on Assets (ROA)'
#     elif type == 'yoy': type = '52 Week Return'
#     elif type == 'q': type = 'Q Ratio'
#     elif type == 'mktcap': type = 'Market Capitalization'
#     elif type == 'ta': type = 'Total Assets'
    
# """

data = pd.merge(left=wrangle.returns(index, measure='roe'),
                right=wrangle.returns(index,measure='roa'),
                on=['Identifier', 'Company Name'], 
                how='inner',
                suffixes=('_ROE', '_ROA'))

data = pd.merge(left=data,
                right=wrangle.returns(index, measure='yoy'),
                on=['Identifier', 'Company Name'], 
                how='inner',
                suffixes=('_ROA','_YOY'))

data = pd.merge(left=data,
                right = wrangle.returns(index, measure='ta'),
                on=['Identifier', 'Company Name'], 
                how='inner',
                suffixes=('_YOY', '_TA'))

data = pd.merge(left=data,
                right = wrangle.returns(index, measure='mktcap'),
                on=['Identifier', 'Company Name'], 
                how='inner',
                suffixes=('_TA', '_MKTCAP'))

data = pd.merge(left=data,
                right = wrangle.returns(index, measure='q'),
                on=['Identifier', 'Company Name'], 
                how='inner',
                suffixes=('_MKTCAP', '_Q'))

data.to_excel(f"{filepath}{filename}",
              sheet_name = 'Financial Returns')