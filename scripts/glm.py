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

# Change this!!!
data = data_finance

# Generalized Linear Models
# Finance - model 
# Regression EQN: ROE_finance ~ ESG_finance + Q_finance

eqn = "ROE ~ ESG + Q_Ratio"
X = data[['ESG', 'Q_Ratio']]
Y = data[['ROE']]

# Strictly positive transformation 
Y = Y - Y.min() + (1e-4)

# Log-transformation 
Y = np.sign(Y) * np.log1p(np.abs(Y))

print (X)
print (Y.min())

glm = smf.glm(eqn, 
              data=data, 
              family=sm.families.Gamma(link=sm.families.links.log())).fit()

data = str(glm.summary())
eqn = "Gamma w log link ROE = ESG + Q"

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print (vif)

shapiro_stat, shapiro_p = stats.shapiro(Y)
print("Shapiro-Wilk Test p-value:", round(shapiro_p, 15))

print (glm.summary())

sm.qqplot(glm.resid_deviance,
            line='45')
plt.title(f'QQ Plot of Gaussian GLM Residuals\n{eqn}')
plt.show()

sns.histplot(glm.resid_deviance, kde=True)
plt.title(f'Histogram of Residuals\n{eqn}')
plt.show()


output_path = config.results_path + 'fin_glm/'
os.makedirs(output_path, exist_ok=True)
output = os.path.join(output_path, eqn)

with open(output, "w") as file: 
    file.write (f"Variance Inflation Factor table\n")
    file.write (str(vif))
    file.write ('\n\n')
    file.write (f"Shapiro-Wilk Normality Test p-value: {round(shapiro_p, 10)}")
    file.write ('\n\n\n')
    file.write (data)

