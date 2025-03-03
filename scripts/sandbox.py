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


def melt_df(df, measure):
    df = df.reset_index()
    
    df.drop(columns=['GICS Industry Name', 'Exchange Name'], inplace=True)
    df.dropna(inplace=True)
    
    df= df.melt(
        id_vars=['Identifier', 'Company Name'],
        var_name='Year',
        value_name=measure.upper()
    )

    return df 

roe = melt_df(wrangle.returns('msci', 'roe'), 'roe')
roe = roe['ROE']

print (f"Sample mean: {roe.sample(n=1000, random_state=1).mean()}")
print (f"Actual mean: {roe.mean()}")

mean = []

def calc_sample_mean(sample_size, no_of_sample_means):
    for i in range(no_of_sample_means):        
        sample_base_salary = roe.sample(n=sample_size)
        sample_mean=sample_base_salary.mean()
        mean.append(sample_mean)
    return mean



# sns.histplot(roe, color='grey')
# plt.show()

mean_2=calc_sample_mean(sample_size=150, no_of_sample_means=5000)
sns.histplot(mean_2, color='b')
plt.show()