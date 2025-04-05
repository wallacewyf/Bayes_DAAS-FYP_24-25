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

# import diagnostic plots and statistical tests
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
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[measure])
            
            data = data.rename(columns={measure: f'log_{measure}_lag_{n_shift}'})
            measure = f'log_{measure}_lag_{n_shift}'

            output_path = config.threshold_log_lag

        # Introducing log1p (Y)
        elif log_transform: 
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            
            data = data.rename(columns={measure: f'log_{measure}'})
            measure = f'log_{measure}'

            output_path = config.threshold_log_lm

        # Introducing n-year lag
        elif n_shift is not None:
            data = data.sort_index()
            data.loc[:, f'{measure}_lag_{n_shift}'] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[f'{measure}_lag_{n_shift}'])

            measure = measure.upper() + f'_lag_{n_shift}'

            output_path = config.threshold_lagged_lm

        else: output_path = config.threshold_basic_lm

    else:
        # Introducing log1p(Y) and n-year lag
        if log_transform and n_shift is not None:
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))
            data[measure] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[measure])

            data = data.rename(columns={measure: f'log_{measure}_lag_{n_shift}'})
            measure = f'log_{measure}_lag_{n_shift}'

            output_path = config.log_lag

        # Introducing log1p (Y)
        elif log_transform: 
            data = data.sort_index()
            data.loc[:, measure] = np.sign(data[measure]) * np.log1p(np.abs(data[measure]))

            data = data.rename(columns={measure: f'log_{measure}'})
            measure = f'log_{measure}'

            output_path = config.log_lm

        # Introducing n-year lag
        elif n_shift is not None:
            data = data.sort_index()
            data.loc[:, f'{measure}_lag_{n_shift}'] = data.groupby(level='Company Name')[[f'{measure}']].shift(n_shift)
            data = data.dropna(subset=[f'{measure}_lag_{n_shift}'])

            measure = measure.upper() + f'_lag_{n_shift}'

            output_path = config.lagged_lm

        else: output_path = config.basic_lm

    # if Empty dataframe -> exit function
    if data.empty: 
        log.warning (f"Empty Dataframe!")
        print (f'WARNING: Check log_file - {config.filename}')
        return

    # Append onto output_path with industry and measure folder
    output_path = os.path.join(output_path, f'{industry}/{measure}/')

    # ESG Parameter Transformation + VIF computation
    # Assign independent variables for regression
    esg = esg.lower()

    if not df.equals(wrangle.all):
        if esg == 'combined':
            eqn = f"{measure} ~ ESG + Q_Ratio"
            output_path += 'ESG/'

            X = data[['ESG','Q_Ratio']]
            vif = stest.vif_calc(data[['ESG','Q_Ratio']])

        elif esg == 'individual':
            eqn = f"{measure} ~ E + S + G + Q_Ratio"
            output_path += 'E_S_G/'

            X = data[['E', 'S', 'G', 'Q_Ratio']]
            vif = stest.vif_calc(data[['E', 'S', 'G', 'Q_Ratio']])

    else:
        vif = None
        if esg == 'combined':
            eqn = f"{measure} ~ ESG"
            output_path += 'ESG/'

            X = data[['ESG']]

        elif esg == 'individual':
            eqn = f"{measure} ~ E + S + G"
            output_path += 'E_S_G/'

            X = data[['E', 'S', 'G']]

    # Predictor Variable - linear regression
    Y = data[[measure]]

    # Verifies if output_path is valid
    os.makedirs(output_path, exist_ok=True)

    # Creates Histogram of Predictor Variables to check distribution 
    sns.histplot(data[measure], bins = 100, kde=True)
    plt.savefig(output_path + f"{measure} Histogram")
    plt.clf()

    # Creates boxplot of Predictor Variables to show outliers
    sns.boxplot(data[measure])
    plt.ylabel(f"{measure} (%)")
    plt.savefig(output_path + f"{measure} Boxplot")
    plt.clf()

    return data, industry, eqn, vif, output_path, X, Y

def linear_reg(df,
               measure = 'roe', 
               esg = 'combined',
               year_threshold=None, 
               log_transform=None, 
               n_shift=None,
               cov_type=None):

    '''
    Linear Regression (Ordinary Least Squares)
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
    Performs linear regression (OLS) based on arguments entered
    '''

    # Initialize dataset
    _, industry, eqn, vif, output_path, X, Y = init_data(df = df, 
                                                  measure = measure.lower(), 
                                                  esg = esg.lower(), 
                                                  year_threshold = year_threshold,
                                                  log_transform = log_transform,
                                                  n_shift = n_shift)
    
    # Extract measure
    measure = eqn.split("~")[0].strip()    

    # Creates baseline
    X = sm.add_constant(X)

    if cov_type is None:
        lm = sm.OLS(Y, X).fit()
    
    else: 
        lm = sm.OLS(Y, X).fit(cov_type=cov_type.upper())

    # Statistical Test values
    shapiro, bp, chi2, aic, bic = stest.diagnostics(predictor_variable = Y,
                                                    model_type = 'lm',
                                                    model = lm)
    
    # Diagnostic Plots
    dplot.export_graphs(model=lm, 
                        model_type='lm', 
                        esg=esg, 
                        measure=measure, 
                        path=output_path)
    
    # Export results into .txt 
    dplot.export_results(summary=lm.summary(), 
                         vif=vif, 
                         industry=industry, 
                         eqn=eqn, 
                         shapiro_p_value=shapiro, 
                         bp_p_value=bp, 
                         chi2_p_value=chi2, 
                         aic=aic, 
                         bic=bic,
                         path=output_path)
    
    return round(lm.llf, 5), round(aic, 5), round(bic, 5)

# Codespace 
# =======================================================
