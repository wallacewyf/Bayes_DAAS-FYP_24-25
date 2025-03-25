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

def model(X, Y, path):
    '''
    Linear Regression Model 
    
    X: independent variables
    Y: dependent / predictor variable
    '''

    # Creates baseline
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit()

    measure = Y.columns.values[0]

    if len(X.columns.values) > 3:
        score = 'E, S, G'
        eqn = f"{measure} ~ E + S + G + Q"

    else: 
        score = 'ESG'
        eqn = f"{measure} ~ ESG + Q"

    residuals = model.resid

    # Diagnostic Plots
    # Residuals Histograms
    sns.histplot(model.resid)
    plt.title(f"Residuals of {score} / {measure} Linear Regression")
    plt.savefig(path + f"{score}_{measure} Residuals Histogram")
    plt.clf()

    # QQ-Plot
    stats.probplot(model.resid, dist='norm', plot=plt)
    plt.title(f"QQ Plot of Residuals of {score} / {measure} Linear Regression")
    plt.savefig(path + f"{score}_{measure} QQ Plot")
    plt.clf()

    # Statistical Tests
    '''
    Shapiro-Wilks Test (Normality)
    --------------------------------------------
    H0: Normally distributed, p >= 0.05
    H1: Not normally distributed, p < 0.05
    '''
    shapiro_stat, shapiro_p = stats.shapiro(Y)

    if shapiro_p < 0.05: shapiro_response = "Fail to reject H0; not normally distributed"
    else: shapiro_response = "Reject H0; normally distributed"
    log.info (f"Shapiro-Wilks check: {shapiro_response}")

    '''
    Pearson Chi-2 Test (Goodness-of-Fit)
    --------------------------------------------
    H0: Good model fit, p >= 0.05
    H1: Bad model fit, p < 0.05
    '''
    observed = np.abs(residuals)
    expected = np.full_like(observed, np.mean(observed))  
    chi2_stat, chi2_p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    if chi2_p_value >= 0.05: pc2_response = 'Fail to reject H0; good fit'
    else: pc2_response = 'Reject H0; bad fit'

    log.info (f"Pearson Chi-2 check: {pc2_response}")

    '''
    Breusch-Pagan test (Heteroscedasticity)
    --------------------------------------------
    H0: Homoscedasticity present, p >= 0.05
    H1: Heteroscedasticity present, p < 0.05
    '''
    bp_test = het_breuschpagan(residuals, model.model.exog)

    bp_test_statistic, bp_test_p_value = bp_test[0], bp_test[1]

    if bp_test_p_value >= 0.05: bp_response = "Fail to reject H0; Homoscedasticity present"
    else: bp_response = "Reject H0; Heteroscedasticity present"

    log.info (f"Breusch-Pagan check: {bp_response}")

    with open(path + f'{eqn}.txt', 'a') as file:
        file.write (str(model.summary()))
        file.write ('\n\n')
        file.write (f"Diagnostic Statistical Tests:\n")
        file.write ('----------------------------------------------------------------------------\n')
        file.write (f"Shapiro-Wilk Normality Test p-value: {shapiro_p}\n")
        file.write (f"Breusch-Pagan Test p-value: {bp_test_p_value}\n")
        file.write (f"Chi-square p-value: {chi2_p_value}\n")
    
    log.info (f'Results exported to {path}\n')

def vif_calc(X):
    '''
    Variance Inflation Factor
    '''

    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF Value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif

def linear_reg(df, measure, type):
    '''
    Basic Linear Regression

    Parameters:

    df = industry data
    measure = ROE / ROA

    Type 1 = ESG / predictor variable
    Type 2 = E,S,G / predictor variable

    '''
    data = df 
    industry = data.index.get_level_values(1).unique()[0]
    measure = measure.upper()

    # Set independent variables to be ESG or E,S,G according to type
    if type == 1:
        X = data[['ESG', 'Q_Ratio']]
        eqn = f"{measure} ~ ESG + Q"

        if len(data.index.get_level_values(2).unique()):
            output_path = config.threshold_basic_lm + f'{industry}/{measure}/ESG/'
        
        else: output_path = config.basic_lm + f'{industry}/{measure}/ESG/'

    elif type == 2:
        X = data[['E', 'S', 'G', "Q_Ratio"]]
        eqn = f"{measure} ~ E + S + G + Q"
        output_path = config.basic_lm + f'{industry}/{measure}/E_S_G/'

    # Checks validity of output path
    os.makedirs(output_path, exist_ok=True)

    # Predictor variable
    Y = data[[f'{measure}']]

    # Check for zero values in X, Y data
    if 0 in X.values:
        if type == 1:
            log.warning (f"ESG / Q_Ratio contains zero values")
        
        else: log.warning ("E, S, G / Q_Ratio contains zero values")
    
    if 0 in Y.values: log.warning (f"{measure} contains zero values")

    '''
    Creates histogram of predictor variable and 
    saves it into results/basic_lm/{type} directory
    '''
    plt.hist(Y, bins = 100)
    plt.title(f"Histogram of {eqn}")
    plt.savefig(output_path + f"{eqn} Histogram")
    plt.clf()

    # Calculate VIF values
    vif = vif_calc(X)
    
    # Write VIF results to file
    with open(output_path + f'{eqn}.txt', "w") as file: 
        file.write (f"GICS Sector: {data.index.get_level_values(1).unique()[0]}\n")
        file.write (f"Regression Equation: {eqn} \n\n")
        file.write (f"Generated: {datetime.datetime.now()} \n\n")
        
        file.write (f"Variance Inflation Factor table\n")
        file.write (str(vif))
        file.write ('\n\n')

    # Creates model
    model(X, Y, output_path)

def lagged_reg(df, measure, type, n):
    '''
    Lagged Linear Regression

    Parameters:

    df = industry data
    measure = ROE / ROA

    Type 1 = ESG / predictor variable
    Type 2 = E,S,G / predictor variable

    n = shift by n year(s)

    '''
    data = df 
    measure = measure.upper()

    # Introducing n-year lag
    # SettingwithCopyWarning needs to be skipped so maybe just replace with 'measure'
    # BUT n-year has to be placed somewhere on the file
    
    data = data.sort_index()
    data.loc[:, f'{measure}_lag_{n}'] = data.groupby(level='Company Name')[[f'{measure}']].shift(n)
    data = data.dropna(subset=[f'{measure}_lag_{n}'])

    industry = data.index.get_level_values(1).unique()[0]
    measure = measure.upper() + f'_lag_{n}'

    # Set independent variables to be ESG or E,S,G according to type
    if type == 1:
        X = data[['ESG', 'Q_Ratio']]
        eqn = f"{measure} ~ ESG + Q"
        output_path = config.lagged_lm + f'{industry}/{measure}/ESG/'

    elif type == 2:
        X = data[['E', 'S', 'G', "Q_Ratio"]]
        eqn = f"{measure} ~ E + S + G + Q"
        output_path = config.lagged_lm + f'{industry}/{measure}/E_S_G/'

    # Checks validity of output path
    os.makedirs(output_path, exist_ok=True)

    # Predictor variable
    Y = data[[f'{measure}']]
    
    # Check for zero values in X, Y data
    if 0 in X.values:
        if type == 1:
            log.warning (f"ESG / Q_Ratio contains zero values")
        
        else: log.warning ("E, S, G / Q_Ratio contains zero values")
    
    if 0 in Y.values: log.warning (f"{measure} contains zero values")

    '''
    Creates histogram of predictor variable and 
    saves it into results/lagged_lm/{type} directory
    '''

    plt.hist(Y, bins = 100)
    plt.title(f"Histogram of {eqn}")
    plt.savefig(output_path + f"{eqn} Histogram")
    plt.clf()

    # Calculate VIF values
    vif = vif_calc(X)

    # Write VIF results to file
    with open(output_path + f'{eqn}.txt', "w") as file: 
        file.write (f"GICS Sector: {data.index.get_level_values(1).unique()[0]}\n")
        file.write (f"Regression Equation: {eqn} \n\n")
        file.write (f"Generated: {datetime.datetime.now()} \n\n")
        
        file.write (f"Variance Inflation Factor table\n")
        file.write (str(vif))
        file.write ('\n\n')

    # Creates model
    model(X, Y, output_path)

def log_linear(df, measure, type):
    '''
    Basic Linear Regression with Log-Transformation on Y

    Parameters:

    df = industry data
    measure = ROE / ROA

    Type 1 = ESG / predictor variable
    Type 2 = E,S,G / predictor variable

    '''
    data = df 
    industry = data.index.get_level_values(1).unique()[0]
    measure = measure.upper()

    # Predictor variable
    data = data.sort_index()
    data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
    data = data.rename(columns={measure: f'log_{measure}'})

    measure = 'log_' + measure
    Y = data[[f'{measure}']]    

    # Set independent variables to be ESG or E,S,G according to type
    if type == 1:
        X = data[['ESG', 'Q_Ratio']]
        eqn = f"{measure} ~ ESG + Q"
        output_path = config.log_lm + f'{industry}/{measure}/ESG/'

    elif type == 2:
        X = data[['E', 'S', 'G', "Q_Ratio"]]
        eqn = f"{measure} ~ E + S + G + Q"
        output_path = config.log_lm + f'{industry}/{measure}/E_S_G/'

    # Checks validity of output path
    os.makedirs(output_path, exist_ok=True)

    # Check for zero values in X, Y data
    if 0 in X.values:
        if type == 1:
            log.warning (f"ESG / Q_Ratio contains zero values")
        
        else: log.warning ("E, S, G / Q_Ratio contains zero values")
    
    if 0 in Y.values: log.warning (f"{measure} contains zero values")

    '''
    Creates histogram of predictor variable and 
    saves it into results/log_lm/{type} directory
    '''
    plt.hist(Y, bins = 100)
    plt.title(f"Histogram of {eqn}")
    plt.savefig(output_path + f"{eqn} Histogram")
    plt.clf()

    # Calculate VIF values
    vif = vif_calc(X)

    # Write VIF results to file
    with open(output_path + f'{eqn}.txt', "w") as file: 
        file.write (f"GICS Sector: {data.index.get_level_values(1).unique()[0]}\n")
        file.write (f"Regression Equation: {eqn} \n\n")
        file.write (f"Generated: {datetime.datetime.now()} \n\n")
        
        file.write (f"Variance Inflation Factor table\n")
        file.write (str(vif))
        file.write ('\n\n')

    # Creates model
    model(X, Y, output_path)

def log_lag(df, measure, type, n):
    '''
    Log-Transformation + Lagged Linear Regression

    Parameters:

    df = industry data
    measure = ROE / ROA

    Type 1 = ESG / predictor variable
    Type 2 = E,S,G / predictor variable

    n = shift by n year(s)

    '''
    data = df 
    industry = data.index.get_level_values(1).unique()[0]
    measure = measure.upper()

    # Introducing n-year lag
    # SettingwithCopyWarning needs to be skipped so maybe just replace with 'measure'
    # BUT n-year has to be placed somewhere on the file

    data = data.sort_index()
    data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
    data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n)
    data = data.dropna(subset=[measure])

    data = data.rename(columns={measure: f'log_{measure}_lag_{n}'})
    measure = f'log_{measure}_lag_{n}'

    # Set independent variables to be ESG or E,S,G according to type
    if type == 1:
        X = data[['ESG', 'Q_Ratio']]
        eqn = f"{measure} ~ ESG + Q"
        output_path = config.log_lag + f'{industry}/{measure}/ESG/'

    elif type == 2:
        X = data[['E', 'S', 'G', "Q_Ratio"]]
        eqn = f"{measure} ~ E + S + G + Q"
        output_path = config.log_lag + f'{industry}/{measure}/E_S_G/'

    # Checks validity of output path
    os.makedirs(output_path, exist_ok=True)

    # Predictor variable
    Y = data[[f'{measure}']]
    
    # Check for zero values in X, Y data
    if 0 in X.values:
        if type == 1:
            log.warning (f"ESG / Q_Ratio contains zero values")
        
        else: log.warning ("E, S, G / Q_Ratio contains zero values")
    
    if 0 in Y.values: log.warning (f"{measure} contains zero values")

    '''
    Creates histogram of predictor variable and 
    saves it into results/log_lag/{type} directory
    '''

    plt.hist(Y, bins = 100)
    plt.title(f"Histogram of {eqn}")
    plt.savefig(output_path + f"{eqn} Histogram")
    plt.clf()

    # Calculate VIF values
    vif = vif_calc(X)

    # Write VIF results to file
    with open(output_path + f'{eqn}.txt', "w") as file: 
        file.write (f"GICS Sector: {data.index.get_level_values(1).unique()[0]}\n")
        file.write (f"Regression Equation: {eqn} \n\n")
        file.write (f"Generated: {datetime.datetime.now()} \n\n")
        
        file.write (f"Variance Inflation Factor table\n")
        file.write (str(vif))
        file.write ('\n\n')

    # Creates model
    model(X, Y, output_path)

def threshold_linear(df, measure, type, year):
    data = df 
    data = data[data.index.get_level_values(2) > year]

    linear_reg(df = data, 
               measure = measure, 
               type = type)
    
def threshold_log_linear(df, measure, type, year):
    data = df 
    data = data[data.index.get_level_values(2) > year]

    log_linear(df = data, 
               measure = measure, 
               type = type)
    
def threshold_lag(df, measure, type, year, n):
    data = df 
    data = data[data.index.get_level_values(2) > year]

    lagged_reg(df = data, 
               measure = measure, 
               type = type, 
               n = n)
    
def threshold_log_lag(df, measure, type, year, n):
    data = df 
    data = data[data.index.get_level_values(2) > year]

    log_lag(df = data, 
               measure = measure, 
               type = type, 
               n = n)
    

# Codespace 
# =======================================================