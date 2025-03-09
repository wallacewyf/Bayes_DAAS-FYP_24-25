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

df = pd.read_excel('/Users/wallace/Documents/code/python/fyp_24/data/GICS_table.xlsx')

print (df.head())
