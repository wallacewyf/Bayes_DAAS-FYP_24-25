# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
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
import config

# initialize logger module 
log = config.logging

def export_graphs(model, 
                  model_type = 'lm',
                  esg='combined', 
                  measure='roe', 
                  path=config.results_path):
    
    '''
    Diagnostic Plots
    ----------------------------------------------------------
    Parameters:
        - model 
        - model_type = LM (default) / GLM
        - esg = combined (default) / individual 
        - measure = roe (default) / roa
        - path = config.results_path (default) 
    ----------------------------------------------------------
    Creates the below diagnostic plots and saves it in the specified path of the model
        - Histogram of Model's Residuals
        - QQ-Plot of Model's Residuals
    '''

    if model_type.lower() == 'lm':
        residuals = model.resid

    elif model_type.lower() == 'glm':
        residuals = model.resid_deviance 

    if esg.lower() == 'combined': score = 'ESG'
    elif esg.lower() == 'individual': score = 'E,S,G'

    # Residuals Histograms
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"Residuals of {score} / {measure}")
    plt.ylabel("Frequency")
    plt.savefig(path + f"Residuals Histogram")
    plt.clf()

    # QQ-Plot
    stats.probplot(residuals, dist='norm', plot=plt)
    plt.title(f"QQ Plot of Residuals of {score} / {measure}")
    plt.savefig(path + f"QQ Plot")
    plt.clf()
                
def export_results(summary,
                   vif,
                   industry,
                   eqn,
                   shapiro_p_value=None, 
                   bp_p_value=None, 
                   chi2_p_value=None,
                   aic=None, 
                   bic=None,
                   transformation_note=None,
                   path=config.results_path):
    
    with open(path + f'{eqn}.txt', 'w') as file: 
        file.write (f"GICS Sector: {industry}\n")
        file.write (f"Regression Equation: {eqn} \n\n")
        file.write (f"Generated: {datetime.datetime.now()} \n\n")

        file.write (str(summary))
        file.write ('\n\n')
        file.write (f"Variance Inflation Factor Table\n")
        file.write ('----------------------------------------------------------------------------\n')
        file.write (str(vif))
        file.write ('\n\n')
        file.write (f"Diagnostic Statistical Tests:\n")
        file.write ('----------------------------------------------------------------------------\n')
        file.write (f"Shapiro-Wilk Normality Test p-value: {shapiro_p_value}\n")
        file.write (f"Breusch-Pagan Test p-value: {bp_p_value}\n")
        file.write (f"Chi-square p-value: {chi2_p_value}\n")
        file.write (f"AIC: {aic}\n")
        file.write (f"BIC: {bic}\n\n")

    # Not sure if this is required
    if transformation_note is not None:
        with open(path + f'{eqn}.txt', 'a') as file:
            file.write (f"Notes:\n")
            file.write ('----------------------------------------------------------------------------\n')
            file.write ('Transformation has occurred on predictor variable!\n')
            file.write (f'{transformation_note}')