# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler
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

# Change this!!!
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB

data = data_tech

# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB
# CHANGE THIS IF YOU'RE NOT DUMB

if data.equals(data_tech):
    output_path = config.results_path + 'tech_glm/ROE/tweedie/'


# Generalized Linear Models
# Finance - model 
# Regression EQN: ROE_finance ~ ESG_finance + Q_finance

def notes():
    '''
    Gaussian - tried and failed 
    
    Tweedie/Gamma - note: has to be strictly positive

    I have no idea why Tweedie and Gamma doesn't work
    (GO FIGURE IT OUT)
    
    '''

# Codespace
# =======================================================
yr_thresh = 2003

if yr_thresh == 2003: 
    filename = f"ROE = ESG + Q"

else:
    filename = f"ROE = ESG + Q post-{yr_thresh}"

data = data[data.index.get_level_values(2) > yr_thresh]
print (data.head())

eqn = "ROE ~ 1"

# =======================================================

# explanatory variable
X = data[['ESG', "Q_Ratio"]]

# # interaction variable
# X['ESG_Q_Ratio'] = X['ESG'] * X['Q_Ratio']

# predictor variable
Y = data[['ROE']]

# Strictly positive transformation 
Y = Y.clip(lower=1e-6)

# Log-transformation 
# Y = np.sign(Y) * np.log(np.abs(Y))

# scaling ESG and Q for Tweedie to work?
X = StandardScaler().fit_transform(X)

# print(X.isnull().sum())
print(np.isnan(X).any())

print (f"THIS IS THE MINIMUM VALUE -------> {Y.min()}")


glm = smf.glm(eqn, 
              data=data, 
              family=sm.families.Tweedie(var_power=1.5))

glm = glm.fit(disp=1)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print (vif)

shapiro_stat, shapiro_p = stats.shapiro(Y)
print("Shapiro-Wilk Test p-value:", round(shapiro_p, 15))

print (glm.summary())

sm.qqplot(glm.resid_deviance,
            line='45')

plt.title(f'QQ Plot of Tweedie at 1.5 Power - GLM Residuals\n{eqn}')
plt.show()

os.makedirs(output_path, exist_ok=True)
output = os.path.join(output_path, filename)

with open(output, "w") as file: 
    file.write (f"GICS Sector: {str(data.index.get_level_values(1).unique()[0])}\n\n")
    file.write (f"Regression equation: {eqn}\n\n")
    file.write (str(glm.summary()))
    file.write ('\n\n\n')
    file.write (f"Shapiro-Wilk Normality Test p-value: {round(shapiro_p, 10)}")
    file.write ('\n\n')
    file.write (f"Variance Inflation Factor table\n")
    file.write (str(vif))


