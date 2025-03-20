# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score                        # R-squared
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels import api as sm 
from statsmodels.formula import api as smf
from scipy import stats

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# other path
import config, wrangle

# initialize logger module 
log = config.logging

data_finance = wrangle.access_data('fin')
data_tech = wrangle.access_data('tech')

data = data_finance

if data.equals(data_finance):
    output_path = config.results_path + 'fin_lm/ROA/'

elif data.equals(data_tech):
    output_path = config.results_path + 'tech_lm/ROA/'

# Linear Regression trial 
# Technology - model 
# Regression EQN: ROE_finance ~ ESG_finance + Q_finance

def notes():
    '''
    Checked 2008 data - different shape from existing shape
    Outliers are real - but linearity to be confirmed

    Tried with and without lag, results are sved in results dir
    {industry}_lm folder
    Previous model used Gaussian    
    '''

def finance_model():
    '''
    Trying threshold cutoff after Kyoto/Paris for Finance    
    
    Verified that ROE ~ ESG is not significant
    Verified that ln(ROE) ~ ESG is significant but negatively correlated
        - even with threshold at year n

    Refer to Notes in wrangle.py
    '''

def tech_model():
    '''
    Verified that linear regression only works if threshold > 2010

    '''
# Codespace
# =======================================================

# yr_thresh = 2010
eqn = f"ROA ~ ESG + Q"

# data = data_finance[data_finance.index.get_level_values(2) > yr_thresh]
print (data.head())

# =======================================================

# explanatory variable
X = data[['ESG', "Q_Ratio"]]

# # interaction variable
# # X['ESG_Q_Ratio'] = X['ESG'] * X['Q_Ratio']

# predictor variable
Y = data[['ROA']]

# Introduce n-year lag (not recommended if shortened time-series analysis)
# # n-year Shift
# n = 1

# data_tech['ROE_Lagged'] = data_tech.groupby(level='Company Name')['ROE'].shift(n)

# This ensures the model only uses rows where lagged values exist
# data_tech = data_tech.dropna(subset=['ROE_Lagged'])

# Step 3: Define X (independent variables) and Y (dependent variable)
# X = data_tech[['ESG', 'Q_Ratio']]
# Y = data_tech[['ROE_Lagged']]  # Use lagged ROE as the dependent variable

# Log-transformation 
# Y = np.sign(Y) * np.log(np.abs(Y))

if 0 in X.values: print ("ESG / Q_Ratio contains zero values")
if 0 in Y.values: print ("ROA contains zero values")

X = sm.add_constant(X)
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF Value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print (f"Variance Inflation Factor table\n")
print (vif)
print ()

reg = sm.OLS(Y,X).fit()
print ()
print ("ESG / ROA Linear Regression Model")
print (reg.summary())

shapiro_stat, shapiro_p = stats.shapiro(Y)

print("Shapiro-Wilk Test p-value:", round(shapiro_p, 15))

residuals = reg.resid
sns.histplot(residuals)
plt.show()

# Save Regression Results to results directory 
data = str(reg.summary())
os.makedirs(output_path, exist_ok=True)
output = os.path.join(output_path, eqn)

with open(output, "w") as file: 
    file.write (f"GICS Sector: {data.index.get_level_values(1).unique()}\n")
    file.write (f"Variance Inflation Factor table\n")
    file.write (str(vif))
    file.write ('\n\n')
    file.write (f"Shapiro-Wilk Normality Test p-value: {round(shapiro_p, 10)}")
    file.write ('\n\n\n')
    file.write (data)
