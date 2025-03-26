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

def vif_calc(X):
    '''
    Variance Inflation Factor

    Checks for multicollinearity among independent variables in the regression equation
    '''

    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF Value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif

def diagnostics(predictor_variable=None, 
                model_type = 'lm', 
                model=None):
    '''
    Statistical Tests for Diagnostics
    ----------------------------------------------------------
    Parameters: 
        - predictor_variable (Y)
        - model_type = LM (default) / GLM
        - model 
    ----------------------------------------------------------
    Current tests:
        - Shapiro-Wilks Test (Normality)
        - Breusch-Pagan Test (Heteroscedasticity)
        - Chi-Square Test (Goodness-of-Fit)

    '''

    shapiro_p_value, shapiro_response = shapiro_wilks(predictor_variable)
    bp_p_value, bp_response = breusch_pagan(model, model_type)
    chi2_p_value, chi2_p_response = chi_square(model, model_type)

    log.info (f"Normality: {shapiro_response}")
    log.info (f"Heteroscedasticity: {bp_response}")
    log.info (f"Goodness-of-Fit: {chi2_p_response}")

    return shapiro_p_value, bp_p_value, chi2_p_value

def shapiro_wilks(predictor_variable):
    '''
    Shapiro-Wilks Test for Normality
    ----------------------------------------------------------
    Parameters:
        - predictor_variable (Y)
    ----------------------------------------------------------
    Interpretation:
        - H0: Not normally distributed; p-value < 0.05
        - H1: Normally distributed; p-value >= 0.05 
    
    '''

    _, shapiro_p = stats.shapiro(predictor_variable)

    if shapiro_p < 0.05: shapiro_response = "Fail to reject H0; not normally distributed"
    else: shapiro_response = "Reject H0; normally distributed"

    return shapiro_p, shapiro_response

def chi_square(model, 
               model_type = 'lm'):
    
    '''
    Pearson Chi-Square Test for Goodness-of-Fit
    ----------------------------------------------------------
    Parameters:
        - model
        - model_type = LM (default) / GLM
    ----------------------------------------------------------
    Interpretation:
        - H0: Good model fit, p >= 0.05
        - H1: Bad model fit, p < 0.05
    '''

    if model_type.lower() == 'lm':
        residuals = model.resid
        observed_values = np.abs(residuals)
        expected_values = np.full_like(observed_values, np.mean(observed_values))

        _, chi2_p_value = stats.chisquare(f_obs=observed_values, 
                                          f_exp=expected_values)

    elif model_type.lower() == 'glm':
        chi2_value = model.pearson_chi2
        residuals = model.df_resid
    
        chi2_p_value = 1 - stats.chi2.cdf(chi2_value, residuals)

    if chi2_p_value >= 0.05: chi2_p_response = 'Fail to reject H0; good fit'
    else: chi2_p_response = 'Reject H0; bad fit'

    return chi2_p_value, chi2_p_response

def breusch_pagan(model,
                  model_type = 'lm'):
    '''
    Breusch-Pagan Test for Heteroscedasticity
    ----------------------------------------------------------
    Parameters:
        - model
        - model_type = GLM / LM (default)
    ----------------------------------------------------------
    Interpretation
        - H0: Homoscedasticity present, p >= 0.05
        - H1: Heteroscedasticity present, p < 0.05

    '''

    if model_type.lower() == 'lm':
        residuals = model.resid

    elif model_type.lower() == 'glm':
        residuals = model.resid_deviance
    
    exog_values = model.model.exog

    bp_test = het_breuschpagan(residuals, exog_values)

    _, bp_p_value, _, _ = bp_test

    if bp_p_value >= 0.05: bp_response = "Fail to reject H0; Homoscedasticity present"
    else: bp_response = "Reject H0; Heteroscedasticity present"

    return bp_p_value, bp_response
