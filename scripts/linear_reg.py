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

finance = wrangle.finance
tech = wrangle.tech
energy = wrangle.energy
industrials = wrangle.industrials
health = wrangle.healthcare

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

measure_type = ['ROE']
data = finance

if data.equals(energy): datatype = 'energy'
elif data.equals(industrials): datatype = 'industrials'
elif data.equals(health): datatype = 'healthcare'
elif data.equals(finance): datatype = 'finance'
elif data.equals(tech): datatype = 'tech'

for measure in measure_type:
    output_path = config.results_path + f'{datatype}_lm/{measure}/E_S_G/'
    os.makedirs(output_path, exist_ok=True)
    
    # Codespace
    # =======================================================

    # yr_thresh = 2010
    eqn = f"{measure} ~ E, S, G + Q"
    # eqn = f"{measure} ~ ESG + Q"


    # data = data_finance[data_finance.index.get_level_values(2) > yr_thresh]
    print (data.head())

    # =======================================================

    # explanatory variable
    X = data[['E', 'S', 'G', "Q_Ratio"]]
    # X = data[['ESG', "Q_Ratio"]]

    # # interaction variable
    # # X['ESG_Q_Ratio'] = X['ESG'] * X['Q_Ratio']

    # predictor variable
    Y = data[[f'{measure}']]

    # Log-transformation 
    # Y = np.sign(Y) * np.log(np.abs(Y))

    # Check if predictor variable is normally distributed
    plt.hist(Y, bins=100)
    plt.xlim(-1,1)
    plt.title(f"Histogram of {eqn}")
    plt.savefig(output_path + f"{eqn} Histogram")
    plt.show()
    plt.clf()

    # Introduce n-year lag (not recommended if shortened time-series analysis)
    # # n-year Shift
    # n = 1

    # data_tech['ROE_Lagged'] = data_tech.groupby(level='Company Name')['ROE'].shift(n)

    # This ensures the model only uses rows where lagged values exist
    # data_tech = data_tech.dropna(subset=['ROE_Lagged'])

    # Step 3: Define X (independent variables) and Y (dependent variable)
    # X = data_tech[['ESG', 'Q_Ratio']]
    # Y = data_tech[['ROE_Lagged']]  # Use lagged ROE as the dependent variable

    if 0 in X.values: print ("E, S, G / Q_Ratio contains zero values")
    # if 0 in X.values: print ("ESG / Q_Ratio contains zero values")
    if 0 in Y.values: print (f"{measure} contains zero values")

    X = sm.add_constant(X)
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF Value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print (f"Variance Inflation Factor table\n")
    print (vif)
    print ()

    reg = sm.OLS(Y,X).fit()
    print ()
    print (f"E, S, G / {measure} Linear Regression Model")
    # print (f"ESG / {measure} Linear Regression Model")
    print (reg.summary())

    shapiro_stat, shapiro_p = stats.shapiro(Y)

    print("Shapiro-Wilk Test p-value:", round(shapiro_p, 15))

    residuals = reg.resid
    sns.histplot(residuals)
    plt.xlim(-1,1)
    plt.title(f"Residuals of E, S, G / {measure} Linear Regression")
    plt.savefig(output_path + f"{measure} Residuals Distribution")
    # plt.show()
    plt.clf()

    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot of Residuals of E, S, G / {measure} Linear Regression")
    # plt.show()
    plt.savefig(output_path + f"QQ Plot {eqn}")
    plt.clf()


    # Save Regression Results to results directory 
    os.makedirs(output_path, exist_ok=True)
    output = os.path.join(output_path, eqn)

    with open(output, "w") as file: 
        file.write (f"GICS Sector: {data.index.get_level_values(1).unique()[0]}\n")
        file.write (f"Regression Equation: {eqn} \n\n")
        file.write (f"Generated: {datetime.datetime.now()} \n\n")
        file.write (f"Variance Inflation Factor table\n")
        file.write (str(vif))
        file.write ('\n\n')
        file.write (f"Shapiro-Wilk Normality Test p-value: {round(shapiro_p, 10)}")
        file.write ('\n\n\n')
        file.write (str(reg.summary()))
