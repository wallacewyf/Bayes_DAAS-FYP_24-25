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

import linear_reg as reg 

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# other path
import config, wrangle

# initialize logger module 
log = config.logging
 
# VIF Calculation available as reg.vif(X)

def init_data(
            df=wrangle.df, 
            measure='roe',
            esg='combined', 
            year_threshold=None,
            log_transform=None,
            n_shift=None
            ):

    '''
    Data Initialization
    ----------------------------------------------------------
    Parameters:
        - df = wrangle.df (default) / wrangle.finance
        - measure = roe (default) / roa
        - esg = combined (default) / individual
        - year_threshold = None (default) / 2004 / 2005 / etc.

          Note: if year_threshold > max(year), max(year) is used.

        - log_transform = None (default) / True / False
        - n_shift = None (default) / 1 / 2 / 3 
          
          Note: if n_shift is longer than max(n), max is used.
                elif n_shift == 0, n_shift = None
    ----------------------------------------------------------
    Initialize data and create histogram of predictor variable (measure)
    and calculate Variance Inflation Factor (VIF)

    Returns wrangled_data, regression_equation, vif, output_path
    '''

    data = df
    measure = measure.upper()

    # n_shift Validation
    # Verify if n_shift within range
    max_shift = len(data.index.get_level_values(2).unique())

    if n_shift is not None:
        if n_shift >= max_shift: 
            n_shift = max_shift - 1

        elif n_shift == 0:
            n_shift = None

    # Introducing threshold on Year
    if year_threshold is not None: 
        # year_threshold Validation
        # Verify if Year within range
        year_range = data.index.get_level_values(2).unique()

        if year_threshold >= year_range.max(): 
            year_threshold = year_range.max()
            data = data[data.index.get_level_values(2) >= year_threshold]
    
        else:
            data = data[data.index.get_level_values(2) > year_threshold]

        # Introducing log1p(Y) and n-year lag
        if log_transform and n_shift is not None:
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[measure])
            
            data = data.rename(columns={measure: f'log_{measure}_lag_{n_shift}'})
            measure = f'log_{measure}_lag_{n_shift}'

            output_path = config.threshold_log_lag_glm

        # Introducing log1p (Y)
        elif log_transform: 
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            
            data = data.rename(columns={measure: f'log_{measure}'})
            measure = f'log_{measure}'

            output_path = config.threshold_log_glm

        # Introducing n-year lag
        elif n_shift is not None:
            data = data.sort_index()
            data.loc[:, f'{measure}_lag_{n_shift}'] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[f'{measure}_lag_{n_shift}'])

            measure = measure.upper() + f'_lag_{n_shift}'

            output_path = config.threshold_lagged_glm

        else: output_path = config.threshold_basic_glm

    else:
        # Introducing log1p(Y) and n-year lag
        if log_transform and n_shift is not None:
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[measure])

            data = data.rename(columns={measure: f'log_{measure}_lag_{n_shift}'})
            measure = f'log_{measure}_lag_{n_shift}'

            output_path = config.log_lag_glm

        # Introducing log1p (Y)
        elif log_transform: 
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))

            data = data.rename(columns={measure: f'log_{measure}'})
            measure = f'log_{measure}'

            output_path = config.log_glm

        # Introducing n-year lag
        elif n_shift is not None:
            data = data.sort_index()
            data.loc[:, f'{measure}_lag_{n_shift}'] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[f'{measure}_lag_{n_shift}'])

            measure = measure.upper() + f'_lag_{n_shift}'

            output_path = config.lagged_glm

        else: output_path = config.basic_glm

    # if Empty dataframe -> exit function
    if data.empty: 
        log.warning (f"Empty Dataframe!")
        print (f'WARNING: Check log_file - {config.filename}')
        return

    # Append onto output_path with measure folder
    output_path = os.path.join(output_path, f'{measure}/')

    # ESG Parameter Transformation + VIF computation
    esg = esg.lower()

    if esg == 'combined':
        eqn = f"{measure} ~ ESG + Q_Ratio"
        output_path += 'ESG/'
        vif = reg.vif_calc(data[['ESG','Q_Ratio']])

    elif esg == 'individual':
        eqn = f"{measure} ~ E + S + G + Q_Ratio"
        output_path += 'E_S_G/'
        vif = reg.vif_calc(data[['E', 'S', 'G', 'Q_Ratio']])

    # Verifies if output_path is valid
    os.makedirs(output_path, exist_ok=True)

    # Creates Histogram of Predictor Variables to check distribution 
    plt.hist(data[measure], bins = 100)
    plt.title(f"Histogram of {eqn}")
    plt.savefig(output_path + f"{measure} Histogram")
    plt.clf()

    return data, eqn, vif, output_path

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
        - Histogram of Predictor Variables
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
    plt.title(f"Residuals of {score} / {measure} Linear Regression")
    plt.savefig(path + f"Residuals Histogram")
    plt.clf()

    # QQ-Plot
    stats.probplot(residuals, dist='norm', plot=plt)
    plt.title(f"QQ Plot of Residuals of {score} / {measure} Linear Regression")
    plt.savefig(path + f"QQ Plot")
    plt.clf()
                
def export_results(summary,
                   vif,
                   industry,
                   eqn,
                   shapiro_p_value=None, 
                   bp_p_value=None, 
                   chi2_p_value=None,
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

def gaussian_glm(df, 
                 measure = 'roe', 
                 esg = 'combined', 
                 year_threshold = None, 
                 log_transform = None, 
                 n_shift = None):
    
    '''
    Gaussian GLMs
    ----------------------------------------------------------
    Parameters:
        - df = wrangle.finance / wrangle.tech
        - measure = ROE (default) / ROA
        - esg = combined (default) / individual
        - year_threshold = None (default) / 2004 / 2020
        - log_transform = None (default) / True / False
        - n_shift = None (default) / 1 / 2 / 3 
          
          Note: if n_shift is longer than max(n), max is used.
                elif n_shift == 0, n_shift = None
    ----------------------------------------------------------
    Performs Gaussian GLM functions based on arguments entered
    '''

    # Initialize dataset
    data, eqn, vif, output_path = init_data(df = df, 
                                            measure = measure.lower(), 
                                            esg = esg.lower(), 
                                            year_threshold=year_threshold,
                                            log_transform=log_transform,
                                            n_shift=n_shift)

        # Extract industry
    industry = data.index.get_level_values(1).unique()[0] if len(data.index.get_level_values(1).unique()) == 1 else None
    measure = eqn.split("~")[0].strip()

    glm = smf.glm(eqn, 
                data=data,
                family=sm.families.Gaussian()).fit()

    shapiro, bp, chi2 = diagnostics(predictor_variable = data[[measure]], 
                                                        model_type = 'glm', 
                                                        model = glm)
    
    export_graphs(model=glm, 
                  model_type='glm', 
                  esg=esg, 
                  measure=measure,
                  path=output_path)
    
    export_results(summary=glm.summary(), 
                   vif=vif, 
                   industry=industry, 
                   eqn=eqn, 
                   shapiro_p_value=shapiro,
                   bp_p_value=bp,
                   chi2_p_value=chi2,
                   path=output_path)


# Notes:
# =======================================================
# Combine basic_ lagged_ log_ log_lag
# It's doable since there are default arguments
# But would have to merged gaussian_init with basic gaussian


# Codespace 
# =======================================================
