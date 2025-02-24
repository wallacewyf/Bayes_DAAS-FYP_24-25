# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score                        # R-squared
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy import stats

import statsmodels.api as sm

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

start = datetime.datetime.now()

# initialize logger module 
log = config.logging

# split between pre/post COVID pandemic / Paris Agreement
# print (esg_sum.loc[:, [2023, 2022, 2021, 2020, 2019, 2017, 2016, 2015, 2014, 2013, 2008, 2004]])

# --------------------------------------------------------------------
# Regression Analysis 
# notes: 
# After discussion with Pietro, run a normal linear regression first by ensuring all returns 
# are normally distributed. 
# If normally distributed, linearity is proven and therefore linear or multivariate regression 
# could then be fitted.

# Second discussion with Pietro, try various tests by excluding outliers and see if the model fits
# Or, slice post 2008 data and since we've verified normally distributed
# Run linear regression on the data

# We could also try applying log transformation on the data to ensure strictly positive values 
# and then reapply the inverse of the transformation after regression model to 
# interpret the predictor

# 20250224 TODO: 
# ESG Change Analysis (see ChatGPT convo history)
# GLM for linear regressions

# Converting DataFrames from wide format to long format (series) for regression analysis
def melt_df(df, measure):
    df = df.reset_index()
    
    df.drop(columns=['GICS Industry Name', 'Exchange Name'], inplace=True)
    df.dropna(inplace=True)
    
    df= df.melt(
        id_vars=['Identifier', 'Company Name'],
        var_name='Year',
        value_name=measure.upper()
    )

    return df 

# Test for Normality
def test_norm(measure, index):
    log.info(f"Running Shapiro-Wilk Test for Normality on {index.upper()}'s {measure.upper()}")

    if measure in ['roe', 'roa', 'mktcap', 'q']: df = wrangle.returns(index, measure)

    df = melt_df(df, measure)

    df.dropna(inplace=True)
    df = df[df['Year'] > 2014]         # omitting 2008 data due to financial crisis

    # taking subset of N = 4500 from data due to large sample size
    shapiro_stat, shapiro_p = stats.shapiro(df[measure.upper()])

    print("Shapiro-Wilk Test statistic:", round(shapiro_stat, 15))
    print("Shapiro-Wilk Test p-value:", round(shapiro_p, 15))
    print ()

    if shapiro_p < 0.05:
        log.info(f"{measure.upper()} is not normally distributed")
        log.info("")
        return False
    
    else: 
        log.info(f"{measure.upper()} is normally distributed")
        log.info('')
        return True

# test_norm('roa', 'msci')
# test_norm('roe', 'msci')
# test_norm('q', 'msci')
# test_norm('mktcap', 'msci')

# --------------------------------------------------------------------------------------------------------------------

# Linear Regression between Measure and ESG + E/S/G
# Included both scikit-learn and statsmodels for more detailed analysis/interpretation
def linear_reg(index, measure):
    log.info (f'{measure.upper()} for {index.upper()} recognized.')
    log.info (f'Melting dataframes into long format...')

    measure_df = melt_df(wrangle.returns(index, measure), measure)
    esg_df = melt_df(wrangle.scores(index, 'esg'), 'esg')
    e_df = melt_df(wrangle.scores(index, 'e'), 'e')
    s_df = melt_df(wrangle.scores(index, 's'), 's')
    g_df = melt_df(wrangle.scores(index, 'g'), 'g')
    
    # TEST FOR EXPLANATORY VARIABLES
    # if we proceed with this, remember to figure out a way to only 
    # select explanatory variables that are not being predicted

    mktcap_df = melt_df(wrangle.returns(index, 'mktcap'), 'mktcap')
    ta_df = melt_df(wrangle.returns(index, 'ta'), 'ta')
    q_df = melt_df(wrangle.returns(index, 'q'), 'q')
    
    log.info (f'Running inner join on dataframes...')
    data = pd.merge(measure_df, esg_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, e_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, s_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, g_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, mktcap_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, ta_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, q_df, on=['Identifier', 'Company Name', 'Year'], how='inner')

    # WARNING: if log-transformation is not applied
    # AND MKTCAP and TA is not removed from the model 
    # P-value does make sense?


    # applying log-transformation to avoid skewness
    data['MKTCAP'] = np.log1p(data['MKTCAP'])
    data['TA'] = np.log1p(data['TA'])

    # if log-transformation is applied, multicolinearity detected for MKTCAP and TA
    # as such, we should remove MKTCAP and TA from model 

    X1 = data[['ESG', 'Q']]
    X2 = data[['E', 'S', 'G', 'Q']]
    Y = data[measure.upper()]

    if 0 in X1.values: log.warning(f'{index.upper()} ESG scores contain zero values.')
    if 0 in X2.values: log.warning(f'{index.upper()} E/S/G scores contain zero values.')
    if 0 in Y.values: log.warning(f'{index.upper()} {measure.upper()} contain zero values.')

    # model for ESG and Measure
    
    log.info (f'Running ESG / {measure.upper()} regression model for {index.upper()}...')
    log.info (f'Running statsmodels regression...')
    X1 = sm.add_constant(X1)

    log.info (f'Testing for multicolinearity...')
    print ('Variance Inflation Factor Computation')
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X1.columns
    vif_data["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]

    print (vif_data)
    print ()

    model = sm.OLS(Y, X1).fit()
    print (f'ESG / {measure.upper()} Linear Regression Model for {index.upper()} \n')
    print (model.summary())
    print ()

    # scikit-learn
    log.info(f'Running scikit-learn regression...')
    X_train, X_test, y_train, y_test = train_test_split(
        X1, 
        Y, 
        test_size=0.2, 
        random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = mean_squared_error(y_test, baseline_pred)

    print (f'Baseline MSE (mean): {baseline_mse}')
    print (f"Mean Squared Error: {mse}")
    print (f"R^2 Score: {r2} \n")

    # --------------------------------------------------------------------------------------------
    # model for E, S, G and Measure
    log.info ('')
    log.info (f'Running E, S, G / {measure.upper()} regression model for {index.upper()}...')
    log.info (f'Running statsmodels regression...')

    X2 = sm.add_constant(X2)

    log.info (f'Testing for multicolinearity...')
    print ('Variance Inflation Factor Computation')
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X2.columns
    vif_data["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]

    print (vif_data)
    print ()

    model = sm.OLS(Y, X2).fit()
    print (f'E, S, G / {measure.upper()} Linear Regression Model for {index.upper()} \n')
    print (model.summary())
    print ()

    log.info(f'Running scikit-learn regression...')
    X_train, X_test, y_train, y_test = train_test_split(
        X2, 
        Y, 
        test_size=0.2, 
        random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = mean_squared_error(y_test, baseline_pred)

    print (f'Baseline MSE (mean): {baseline_mse}')
    print (f"Mean Squared Error: {mse}")
    print (f"R^2 Score: {r2} \n")
    print ()

# Linear Regression on MSCI's measures
linear_reg(index = 'msci', measure = 'roe')
# linear_reg(index = 'msci', measure = 'roa')
# linear_reg(index = 'msci', measure = 'mktcap')
# linear_reg(index = 'msci', measure = 'q')

# Lagged Regression
def lagged_reg(index, measure, max_lag):
    for lag in range (1, max_lag+1):
        log.info (f'{measure.upper()} for {index.upper()} with {max_lag} lag period recognized.')
        
        log.info (f'Melting dataframes into long format...')
        measure_df = melt_df(wrangle.returns(index, measure), measure)
        esg_df = melt_df(wrangle.scores(index, 'esg'), 'esg')
        e_df   = melt_df(wrangle.scores(index, 'e'), 'e')
        s_df   = melt_df(wrangle.scores(index, 's'), 's')
        g_df   = melt_df(wrangle.scores(index, 'g'), 'g')

        log.info (f'Running inner join on dataframes...')
        data = pd.merge(measure_df, esg_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
        data = pd.merge(data, e_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
        data = pd.merge(data, s_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
        data = pd.merge(data, g_df, on=['Identifier', 'Company Name', 'Year'], how='inner')

        data.sort_values(by=['Identifier', 'Year'], inplace=True)

        # shifted values by x year downwards (i.e. 2004 values reassigned to 2005)
        log.info (f'Running regression with {lag}-year lag...')
        for cat in [measure.upper(), 'ESG', 'E', 'S', 'G']:
            data[cat] = data.groupby('Identifier')[cat].shift(lag)

        data.dropna(inplace=True)

        X1 = data[['ESG']]
        X2 = data[['E', 'S', 'G']]
        Y = data[measure.upper()]

        if 0 in X1.values: log.warning(f'{index.upper()} ESG scores contain zero values.')
        if 0 in X2.values: log.warning(f'{index.upper()} E/S/G scores contain zero values.')
        if 0 in Y.values: log.warning(f'{index.upper()} {measure.upper()} contain zero values.')

        # model for ESG and Measure
        
        log.info (f'Running ESG / {measure.upper()} regression model for {index.upper()}...')
        log.info (f'Running statsmodels regression...')
        X1 = sm.add_constant(X1)
        model = sm.OLS(Y, X1).fit()
        print (f'ESG / {measure.upper()} Linear Regression Model for {index.upper()} with {lag}-year lag \n')
        print (model.summary())
        print ()

        X_train, X_test, y_train, y_test = train_test_split(
            X1, 
            Y, 
            test_size=0.2, 
            random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        baseline_pred = np.full_like(y_test, y_train.mean())
        baseline_mse = mean_squared_error(y_test, baseline_pred)

        print (f'Baseline MSE (mean): {baseline_mse}')
        print ("Mean Squared Error:", mse)
        print ("R^2 Score:", r2)
        print ('\n')

        # --------------------------------------------------------------------------------------------
        # model for E, S, G and Measure

        log.info ('')
        log.info (f'Running E, S, G / {measure.upper()} regression model for {index.upper()}...')
        log.info (f'Running statsmodels regression...')

        X2 = sm.add_constant(X2)
        model = sm.OLS(Y, X2).fit()
        print (f'E, S, G / {measure.upper()} Linear Regression Model for {index.upper()} with {lag}-year lag \n')
        print (model.summary())
        print ()

        X_train, X_test, y_train, y_test = train_test_split(
            X2, 
            Y, 
            test_size=0.2, 
            random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        baseline_pred = np.full_like(y_test, y_train.mean())
        baseline_mse = mean_squared_error(y_test, baseline_pred)

        print (f'Baseline MSE (mean): {baseline_mse}')
        print ("Mean Squared Error:", mse)
        print ("R^2 Score:", r2)
        print ('\n')

        residuals = y_test - y_pred

# lagged_reg(index = 'msci', measure = 'roe', max_lag = 5)

end = datetime.datetime.now()

print (f'\n\nTime taken: {end - start}')


# TODO: Correlation, Relationship, Regression, Cluster Analysis 
#       (cross-compare with methods on cited literatures on types of analysis used)
