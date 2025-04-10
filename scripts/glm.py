# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from sklearn.preprocessing import StandardScaler        # Inverse Gaussian
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels import api as sm 
from statsmodels.formula import api as smf
from scipy import stats

# import diagnostic plots, statistical tests
import statistical_tests as stest
import diagnostic_plots as dplot

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# other path
import config, wrangle

# initialize logger module 
log = config.logging

def init_data(
            df=wrangle.df, 
            measure='roe',
            esg='combined', 
            year_threshold=None,
            log_transform=None,
            n_shift=None,
            glm_type='Gaussian'
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
    Initialize data and create the below diagnostics plots:
    - Histogram of predictor variable (measure)
    - Boxplot of Predictor Variables

    and computes Variance Inflation Factor (VIF)

    Returns wrangled_data, industry, regression_equation, vif, output_path
    '''

    data = df
    measure = measure.upper()
    industry = data.index.get_level_values(1).unique()[0] if len(data.index.get_level_values(1).unique()) == 1 else 'All Industries'

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
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
            data[measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[measure])
            
            data = data.rename(columns={measure: f'log_{measure}_lag_{n_shift}'})
            measure = f'log_{measure}_lag_{n_shift}'

            output_path = config.threshold_log_lag_glm

        # Introducing log1p (Y)
        elif log_transform: 
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
            data[measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            
            data = data.rename(columns={measure: f'log_{measure}'})
            measure = f'log_{measure}'

            output_path = config.threshold_log_glm

        # Introducing n-year lag
        elif n_shift is not None:
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
            data.loc[:, f'{measure}_lag_{n_shift}'] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[f'{measure}_lag_{n_shift}'])

            measure = measure.upper() + f'_lag_{n_shift}'

            output_path = config.threshold_lagged_glm

        else: output_path = config.threshold_basic_glm

    else:
        # Introducing log1p(Y) and n-year lag
        if log_transform and n_shift is not None:
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
            data[measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[measure])

            data = data.rename(columns={measure: f'log_{measure}_lag_{n_shift}'})
            measure = f'log_{measure}_lag_{n_shift}'

            output_path = config.log_lag_glm

        # Introducing log1p (Y)
        elif log_transform: 
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
            data[measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))

            data = data.rename(columns={measure: f'log_{measure}'})
            measure = f'log_{measure}'

            output_path = config.log_glm

        # Introducing n-year lag
        elif n_shift is not None:
            data = data.sort_index(level=["Year", 'Company Name', 'GICS Sector Name'],
                                   ascending=[True, True, True])
            
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

    # Append onto output_path with GLM type and industry and measure folder
    if glm_type.lower() == 'inversegaussian' or glm_type.lower() == 'inverse gaussian':
        output_path = os.path.join(output_path, "Inverse Gaussian/")

    elif glm_type.lower() == 'gamma':
        output_path = os.path.join(output_path, "Gamma/")

    elif glm_type.lower() == 'tweedie':
        output_path = os.path.join(output_path, "Tweedie/")

    else: output_path = os.path.join(output_path, "Gaussian/")
    
    output_path = os.path.join(output_path, f'{industry}/{measure}/')

    # ESG Parameter Transformation + VIF computation
    esg = esg.lower()

    if esg == 'combined':
        eqn = f"{measure} ~ ESG + Q_Ratio"
        output_path += 'ESG/'
        vif = stest.vif_calc(data[['ESG','Q_Ratio']])

    elif esg == 'individual':
        eqn = f"{measure} ~ E + S + G + Q_Ratio"
        output_path += 'E_S_G/'
        vif = stest.vif_calc(data[['E', 'S', 'G', 'Q_Ratio']])

    # Verifies if output_path is valid
    os.makedirs(output_path, exist_ok=True)

    # Creates Histogram of Predictor Variables to check distribution 
    sns.histplot(data[measure], kde=True, bins = 100)
    plt.xlabel(f"{measure} (%)")
    plt.ylabel("Frequency")
    plt.savefig(output_path + f"{measure} Histogram")
    plt.clf()

    # Creates boxplot of Predictor Variables to show outliers
    sns.boxplot(data[measure])
    plt.ylabel(f"{measure} (%)")
    plt.savefig(output_path + f"{measure} Boxplot")
    plt.clf()

    return data, industry, eqn, vif, output_path

def gaussian_glm(df, 
                 measure = 'roe', 
                 esg = 'combined', 
                 year_threshold = None, 
                 log_transform = None, 
                 n_shift = None,
                 link = None):
    
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
        - link = None (defaults as Identity) / Log / Inverse
    ----------------------------------------------------------
    Performs Gaussian GLM based on arguments entered
    '''

    # Initialize dataset
    data, industry, eqn, vif, output_path = init_data(df = df, 
                                                      measure = measure.lower(), 
                                                      esg = esg.lower(), 
                                                      year_threshold=year_threshold, 
                                                      log_transform=log_transform, 
                                                      n_shift=n_shift,
                                                      glm_type='Gaussian')

    # Extract measure
    measure = eqn.split("~")[0].strip()

    # Link functions 
    if link is None:
        glm = smf.glm(eqn, 
                    data=data, 
                    family=sm.families.Gaussian()).fit()
    
    elif link.lower() == 'inverse':
        glm = smf.glm(eqn, 
                    data=data, 
                    family=sm.families.Gaussian(link=sm.genmod.families.links.InversePower())).fit()
        
    elif link.lower() == 'log':
        glm = smf.glm(eqn, 
                      data = data, 
                      family = sm.families.Gaussian(link=sm.genmod.families.links.Log())).fit()

    # Statistical Tests
    shapiro, bp, chi2, aic, bic = stest.diagnostics(predictor_variable = data[[measure]], 
                                                    model_type = 'glm', 
                                                    model = glm)
    # Diagnostic Plots
    dplot.export_graphs(model=glm, 
                        model_type='glm', 
                        path=output_path)
    
    # Export results to directory
    dplot.export_results(summary=glm.summary(), 
                         vif=vif, 
                         industry=industry, 
                         eqn=eqn, 
                         shapiro_p_value=shapiro, 
                         bp_p_value=bp, 
                         chi2_p_value=chi2,
                         aic=aic,
                         bic=bic, 
                         path=output_path)
    
    return round(glm.llf, 5), round(aic, 5), round(bic, 5)

def inv_gaussian(df, 
                 measure = 'roe', 
                 esg = 'combined', 
                 year_threshold = None, 
                 log_transform = None, 
                 n_shift = None,
                 link = 'Log'):
        
    '''
    Inverse Gaussian GLMs
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
        - link = Log (default) / None / Inverse
    ----------------------------------------------------------
    Performs Inverse Gaussian GLM based on arguments entered

    Note: strictly positive transformation added here to fulfill
          Inverse Gaussian GLM pre-condition

    '''

    # Initialize dataset
    data, industry, eqn, vif, output_path = init_data(df = df, 
                                                      measure = measure.lower(), 
                                                      esg = esg.lower(), 
                                                      year_threshold=year_threshold, 
                                                      log_transform=log_transform, 
                                                      n_shift=n_shift,
                                                      glm_type='Inverse Gaussian')

    # Extract measure
    measure = eqn.split("~")[0].strip()

    # Positive transformation for Inverse Gaussian only
    data = data.sort_index(ascending=True)
    
    data.loc[:, measure] = data[measure] - data[measure].min() + (1e-4)
    inverse_value = (data[measure].min() + (1e-4))

    # Link functions
    try:
        if link is None:
            log.warning (f"Inverse Gaussian with no link function does not work!")

            glm = smf.glm(eqn, 
                        data=data, 
                        family=sm.families.InverseGaussian()).fit()
            
            return
        
        elif link.lower() == 'inverse':
            log.warning (f"Inverse Gaussian with Inverse Power link function does not work!")

            glm = smf.glm(eqn, 
                        data=data, 
                        family=sm.families.InverseGaussian(link=sm.genmod.families.links.InversePower())).fit()
            
            return
        
        elif link.lower() == 'log':
            glm = smf.glm(eqn, 
                        data = data, 
                        family = sm.families.InverseGaussian(link=sm.genmod.families.links.Log())).fit()
        
            transformation_note = f"Log-link function called -> Interpret coefficient as exp(coef)! \n"
            transformation_note += f"Strictly positive transformation with {inverse_value} \n\n"

        # Statistical Tests
        shapiro, bp, chi2, aic, bic = stest.diagnostics(predictor_variable = data[[measure]], 
                                                        model_type = 'glm', 
                                                        model = glm)
        
        # Diagnostic Plots
        dplot.export_graphs(model=glm, 
                            model_type='glm', 
                            path=output_path)
        
        # Export results to directory
        dplot.export_results(summary=glm.summary(), 
                            vif=vif, 
                            industry=industry, 
                            eqn=eqn, 
                            shapiro_p_value=shapiro, 
                            bp_p_value=bp, 
                            chi2_p_value=chi2,
                            aic=aic,
                            bic=bic,
                            transformation_note=transformation_note,
                            path=output_path)
        
        return round(glm.llf, 5), round(aic, 5), round(bic, 5)

    except ValueError as error_msg:
        log.warning (f"ValueError: {error_msg}")

        print(f"ValueError encountered: {error_msg}")

def gamma_glm(df, 
                measure = 'roe', 
                esg = 'combined', 
                year_threshold = None, 
                log_transform = None, 
                n_shift = None,
                link = 'Log'):
        
    '''
    Gamma GLMs
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
        - link = Log (default) / None 
    ----------------------------------------------------------
    Performs Gamma GLM based on arguments entered

    Note: strictly positive transformation added here to fulfill
          Gamma GLM pre-condition. Since Gamma GLMs has the same 
          conditions as Inverse Gaussian, we mirror the inv_gaussian() 
          function.

    '''


    # Initialize dataset
    data, industry, eqn, vif, output_path = init_data(df = df, 
                                                      measure = measure.lower(), 
                                                      esg = esg.lower(), 
                                                      year_threshold=year_threshold, 
                                                      log_transform=log_transform, 
                                                      n_shift=n_shift,
                                                      glm_type='Gamma')

    # Extract measure
    measure = eqn.split("~")[0].strip()

    # Link functions
    try:
        if link is None:
            log.warning (f"Gamma with no link function does not work!")

            glm = smf.glm(eqn, 
                        data=data, 
                        family=sm.families.Gamma()).fit()
            
            return
        
        elif link.lower() == 'log':
            glm = smf.glm(eqn, 
                        data = data, 
                        family = sm.families.Gamma(link=sm.genmod.families.links.Log())).fit()
            
            transformation_note = f"Log-link function called -> Interpret coefficient as exp(coef)! \n"

        # Statistical Tests
        shapiro, bp, chi2, aic, bic = stest.diagnostics(predictor_variable = data[[measure]], 
                                                        model_type = 'glm', 
                                                        model = glm)
        
        # Diagnostic Plots
        dplot.export_graphs(model=glm, 
                            model_type='glm', 
                            path=output_path)
        
        # Export results to directory
        dplot.export_results(summary=glm.summary(), 
                            vif=vif, 
                            industry=industry, 
                            eqn=eqn, 
                            shapiro_p_value=shapiro, 
                            bp_p_value=bp, 
                            chi2_p_value=chi2,
                            aic=aic,
                            bic=bic,
                            path=output_path)
        
        return round(glm.llf, 5), round(aic, 5), round(bic, 5)

    except ValueError as error_msg:
        log.warning (f"ValueError: {error_msg}")

        print(f"ValueError encountered: {error_msg}")

def tweedie_glm(df, 
                 measure = 'roe', 
                 esg = 'combined', 
                 year_threshold = None, 
                 log_transform = None, 
                 n_shift = None,
                 link = None,
                 var_power = None):
    
    '''
    Tweedie GLMs
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
        - link = None (defaults as Log) / Identity
        - var_power = None / 1 (Poisson) / 1.5 (Tweedie) / 2 (Gamma)
    ----------------------------------------------------------
    Performs Tweedie GLM based on arguments entered
    '''

    # Initialize dataset
    data, industry, eqn, vif, output_path = init_data(df = df, 
                                                      measure = measure.lower(), 
                                                      esg = esg.lower(), 
                                                      year_threshold=year_threshold, 
                                                      log_transform=log_transform, 
                                                      n_shift=n_shift,
                                                      glm_type = 'Tweedie')

    # Extract measure
    measure = eqn.split("~")[0].strip()

    # Link functions 
    if var_power is None:
        if link is None:
            glm = smf.glm(eqn, 
                        data=data, 
                        family=sm.families.Tweedie()).fit()
        
        elif link.lower() == 'identity':
            glm = smf.glm(eqn, 
                        data = data, 
                        family = sm.families.Tweedie(link=sm.genmod.families.links.Identity())).fit()

    else:
        if link is None:
            glm = smf.glm(eqn, 
                        data=data, 
                        family=sm.families.Tweedie(var_power=var_power)).fit()
        
        elif link.lower() == 'identity':
            glm = smf.glm(eqn, 
                        data = data, 
                        family = sm.families.Tweedie(link=sm.genmod.families.links.Identity(),
                                                     var_power=var_power)).fit()
    # Statistical Tests
    shapiro, bp, chi2, aic, bic = stest.diagnostics(predictor_variable = data[[measure]], 
                                                    model_type = 'glm', 
                                                    model = glm)
    
    # Diagnostic Plots
    dplot.export_graphs(model=glm, 
                        model_type='glm', 
                        path=output_path)
    
    # Export results to directory
    dplot.export_results(summary=glm.summary(), 
                         vif=vif, 
                         industry=industry, 
                         eqn=eqn, 
                         shapiro_p_value=shapiro, 
                         bp_p_value=bp, 
                         chi2_p_value=chi2, 
                         aic=aic,
                         bic=bic,
                         path=output_path)
    
    return round(glm.llf, 5), round(aic, 5), round(bic, 5)
    
# Codespace 
# =======================================================