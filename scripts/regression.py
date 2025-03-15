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

# config path
import config, wrangle

# initialize logger module 
log = config.logging

# Notes
# """
# work on visualizing residuals to prove why gaussian glm is best fit
# visuals are to be added into fyp doc
# to prove confirmation on goodness of fit
# maybe add a for loop to retrieve max log-likelihood
# to deem best fit and export results?
# but for consistency it might be best to stick to one model
# and quote limitations in paper

# NEW: DS said to just do visualization.py instead of visualizing residuals
# only visualise them if additional word count quota
# """


# --------------------------------------------------------------------
# Regression Analysis 

# Converting DataFrames from wide format to long format (series) for regression analysis
def melt_df(df, measure):
    log.info (f"Melting {measure.upper()} dataframe into long format...")
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
# We've confirmed that the returns aren't normally distributed
def test_norm(measure, index):
    log.info(f"Running Shapiro-Wilk Test for Normality on {index.upper()}'s {measure.upper()}")

    df = melt_df(wrangle.returns(index, measure), measure)
    df.dropna(inplace=True)

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

    # try lag variable to check norm dist

    data.sort_values(by=['Identifier', 'Year'], inplace=True)

    for cat in [measure.upper(), 'ESG', 'E', 'S', 'G']:
        data[cat] = data.groupby('Identifier')[cat].shift(1)

    data.dropna(inplace=True)
    
    print (f"Before transformed: {data['ROE'].iloc[385]}")
    print ()

    log.info (f'Log-transforming ROE data...')
    data['ROE'] = np.sign(data['ROE']) * np.log1p(np.abs(data['ROE']))

    print (f"Log-transformed = {(data['ROE'].iloc[385])}")

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
    
    # test for normality after log transformation
    stat, p = stats.shapiro(model.resid)

    print(f'Shapiro-Wilk test statistic: {stat}, p-value: {p}')
    
    if p > 0.05:
        print("Residuals are normally distributed")
    else:
        print("Residuals are NOT normally distributed")

    stat, p = stats.kstest(model.resid, 'norm')
    print(f'KS test statistic: {stat}, p-value: {p}')

    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residual vs. Fitted Plot")
    plt.show()

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

    # # --------------------------------------------------------------------------------------------
    # # model for E, S, G and Measure
    # log.info ('')
    # log.info (f'Running E, S, G / {measure.upper()} regression model for {index.upper()}...')
    # log.info (f'Running statsmodels regression...')

    # X2 = sm.add_constant(X2)

    # log.info (f'Testing for multicolinearity...')
    # print ('Variance Inflation Factor Computation')
    # vif_data = pd.DataFrame()
    # vif_data["Variable"] = X2.columns
    # vif_data["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]

    # print (vif_data)
    # print ()

    # model = sm.OLS(Y, X2).fit()
    # print (f'E, S, G / {measure.upper()} Linear Regression Model for {index.upper()} \n')
    # print (model.summary())
    # print ()

    # log.info(f'Running scikit-learn regression...')
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X2, 
    #     Y, 
    #     test_size=0.2, 
    #     random_state=0)

    # model = LinearRegression()
    # model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)

    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # baseline_pred = np.full_like(y_test, y_train.mean())
    # baseline_mse = mean_squared_error(y_test, baseline_pred)

    # print (f'Baseline MSE (mean): {baseline_mse}')
    # print (f"Mean Squared Error: {mse}")
    # print (f"R^2 Score: {r2} \n")
    # print ()

# Linear Regression on MSCI's measures
# linear_reg(index = 'msci', measure = 'roe')
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

# Generalized Linear Model

def gamma_glm():
    # constant e
    eps = 1e-4

    measure_df = melt_df(wrangle.returns('MSCI', 'roe'), 'roe')
    esg_df = melt_df(wrangle.scores('MSCI', 'esg'), 'esg')
    e_df = melt_df(wrangle.scores('MSCI', 'e'), 'e')
    s_df = melt_df(wrangle.scores('MSCI', 's'), 's')
    g_df = melt_df(wrangle.scores('MSCI', 'g'), 'g')

    # TEST FOR EXPLANATORY VARIABLES
    # if we proceed with this, remember to figure out a way to only 
    # select explanatory variables that are not being predicted

    mktcap_df = melt_df(wrangle.returns('MSCI', 'mktcap'), 'mktcap')
    ta_df = melt_df(wrangle.returns('MSCI', 'ta'), 'ta')
    q_df = melt_df(wrangle.returns('MSCI', 'q'), 'q')

    data = pd.merge(measure_df, esg_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, e_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, s_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, g_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, mktcap_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, ta_df, on=['Identifier', 'Company Name', 'Year'], how='inner')
    data = pd.merge(data, q_df, on=['Identifier', 'Company Name', 'Year'], how='inner')

    data['ESG'] = data['ESG'].shift(1)
    data.dropna(inplace=True)

    print (f"pre shift = min val: {data['ROE'].min()}, max val: {data['ROE'].max()}")

    data['ROE'] = data['ROE'] - data['ROE'].min() + eps

    print (f"post shift = min val: {data['ROE'].min()}, max val: {data['ROE'].max()}")

    X = data[['ESG', 'Q']]
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)

    # scaler = StandardScaler()
    # data[['ESG', 'Q']] = scaler.fit_transform(data[['ESG', 'Q']])

    data['crisis_year'] = (data['Year'] == 2008) | (data['Year'] == 2020)

    print (data['crisis_year'][data['crisis_year'] == True])

    # regression eqn 
    gamma_glm = smf.glm("ROE ~ ESG + Q + crisis_year", 
                        data=data,
                        family=sm.families.Gamma(link=sm.genmod.families.links.Log())).fit()

    print (gamma_glm.summary())
    print (round(gamma_glm.aic),5)
    print (round(gamma_glm.bic),5)
    print ()

    # inv_gauss_glm = smf.glm("ROE ~ ESG + Q", 
    #                         data=data,
    #                         family=sm.families.InverseGaussian(link=sm.genmod.families.links.Log())).fit()
    
    # print (inv_gauss_glm.summary())
    
# gamma_glm()

def gaussian_glm(index, measure, scores):    
    # transforming dataframes into long format
    measure_df = melt_df(wrangle.returns(index, measure), measure)
    esg_df = melt_df(wrangle.scores(index, 'esg'), 'esg')
    e_df = melt_df(wrangle.scores(index, 'e'), 'e')
    s_df = melt_df(wrangle.scores(index, 's'), 's')
    g_df = melt_df(wrangle.scores(index, 'g'), 'g')
    q_df = melt_df(wrangle.returns(index, 'q'), 'q')

    roa_df = melt_df(wrangle.returns(index, 'roa'), 'roa')

    if not scores: 
        scores = ['E','S','G']

        data = pd.merge(left=measure_df, 
                        right=e_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')

        data = pd.merge(left=data, 
                        right=s_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')

        data = pd.merge(left=data, 
                        right=g_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')

        data = pd.merge(left=data, 
                        right=q_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')

    else: 
        scores = 'ESG'

        # merging to ensure PK on table (removing duplicates)
        data = pd.merge(left=measure_df, 
                        right=esg_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')

        data = pd.merge(left=data, 
                        right=q_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')
        
        data = pd.merge(left=data, 
                        right=roa_df, 
                        on=['Identifier', 'Company Name', 'Year'], 
                        how='inner')
        
    if type(scores) == list:
        log.info(f"Running GLM for {index.upper()} and {measure.upper()} for E,S,G individual pillars")

        # for each_score in scores: 
        #     log.info(f"Adding 1-year time lag on {each_score} data")
        #     data[each_score.upper()] = data[each_score.upper()].shift(1)
        
        data.dropna(inplace=True)
        
        scores = 'E,S,G'
        X = data[['E', 'S', 'G', 'Q']]

        eqn = f"{measure.upper()} ~ E + S + G + Q + crisis_year"

        # scaler = StandardScaler()
        # data[['E', 'S', 'G', 'Q']] = scaler.fit_transform(data[['E', 'S', 'G', 'Q']])

    else:
        log.info (f"Running GLM for {index.upper()} and {measure.upper()} for {scores} scores")

        # same - before/after comparison but concl market reaction is immediate and p-val > 0.05
        # inserting 1-year time lag on ESG data
        # log.info (f"Adding 1-year time lag on {scores.upper()} data")
        # data['ESG'] = data['ESG'].shift(1)

        # removing n-1 year ESG data
        data.dropna(inplace=True)

        X = data[['ESG', 'Q', 'ROA']]
        eqn = f"{measure.upper()} ~ ESG + Q + ROA + crisis_year"

        # to do a before/after comparison - concl: not needed
        # scaler = StandardScaler()
        # data[['ESG', 'Q']] = scaler.fit_transform(data[['ESG', 'Q']])

    log.info (f"old: min val: {round(data[measure.upper()].min(), 5)}, max val: {round(data[measure.upper()].max(), 5)}")
    log.info ('')

    # minimum value set as 0.00001 (1e-4)
    # transforms to strictly positive
    eps = 1e-4
    data[measure.upper()] = data[measure.upper()] - data[measure.upper()].min() + eps

    # log (1+x) transformation
    data[measure.upper()] = np.sign(data[measure.upper()]) * np.log1p(np.abs(data[measure.upper()]))

    '''
    Yeo-Johnson Transformation 
        worse than log(1+p) transformation
    '''
    # pt = PowerTransformer('yeo-johnson')
    # data[measure.upper()] = pt.fit_transform(data[measure.upper()].values.reshape(-1,1))

    '''
    Shapiro-Wilks Normality Test

    H0: Not normally distributed; p-value < 0.05
    H1: Normally distributed; p-value >= 0.05
    '''
    shapiro_stat, shapiro_p = stats.shapiro(data[measure.upper()])

    print("Shapiro-Wilk Test statistic:", round(shapiro_stat, 15))
    print("Shapiro-Wilk Test p-value:", round(shapiro_p, 15))
    print ()

    if shapiro_p < 0.05: shapiro_res = "Fail to reject H0; not normally distributed"
    else: shapiro_res = "Reject H0; normally distributed"
    
    '''
    dummy variable for crisis year to model them separately
    crisis year = 2008, 2020

    2008: financial crisis
    2020: COVID-19 pandemic

    too many outliers in the 2 years
    '''
    
    data['crisis_year'] = (data['Year'] == 2008) | (data['Year'] == 2020)

    # calculating variance inflation factor to test for multicollinearity
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Gaussian regression eqn 
    gaussian_glm = smf.glm(eqn, 
                        data=data,
                        family=sm.families.Gaussian()).fit(
                            # cov_type='HC3'
                        )
    
    # check for heteroscedasticity
    # diagnostic test - Breusch-Pagan test
    '''
    H0: Homoscedasticity present, p >= 0.05
    H1: Heteroscedasticity present, p < 0.05
    '''
    bp_test = het_breuschpagan(gaussian_glm.resid_deviance, gaussian_glm.model.exog)
    
    bp_test_statistic, bp_test_p_value, _, _ = bp_test

    if bp_test_p_value >= 0.05: bp_response = "Fail to reject H0; Homoscedasticity present"
    else: bp_response = "Reject H0; Heteroscedasticity present"

    # calculate pearson-chi2 p-value
    pearson_chi2 = gaussian_glm.pearson_chi2
    res = gaussian_glm.df_resid
    p_val = 1 - stats.chi2.cdf(pearson_chi2, res)

    sm.qqplot(gaussian_glm.resid_deviance,
              line='45')
    plt.title('QQ Plot of Gaussian GLM Residuals')
    plt.show()

    sns.histplot(gaussian_glm.resid_deviance, kde=True)
    plt.title('Histogram of Residuals')
    plt.show()

    # verify if good fit
    '''
    Pearson Chi-2 Test
    H0: Good model fit, p >= 0.05
    H1: Bad model fit, p < 0.05
    '''

    if p_val >= 0.05: pc_response = 'Fail to reject H0; good fit'
    else: pc_response = 'Reject H0; bad fit'

    log.info (f"Pearson Chi-2 check: {pc_response}")
    log.info ('Refer to regression.py for code')
    log.info ('')

    # export result to file
    output_path = f"{config.glm_path}/{index.upper()}/"

    os.makedirs(output_path,
                exist_ok = True)

    filename = f"{measure.upper()} ~ {scores.upper()} GLM results.txt"
    output_path = os.path.join(output_path, filename)

    with open (output_path, 'w') as result:
        result.write(f'Variance Inflation Factor for {scores} / {measure.upper()} \n')
        result.write(f"{str(vif_data)} \n\n")
        result.write(f"Regression Equation: {eqn} \n\n")
        result.write(str(gaussian_glm.summary()))
        result.write('\n\n')
        result.write(f"Shapiro-Wilks p-value: {round(shapiro_p, 5)}\n{shapiro_res}\n\n")
        result.write(f"Breusch-Pagan p-value: {round(bp_test_p_value, 5)}\n{bp_response}\n\n")
        result.write(f"Pearson Chi-2 p-value: {round(p_val, 5)}\n{pc_response}\n\n")
        result.write(f"AIC Value: {str(round(gaussian_glm.aic, 5))}\n")
        result.write(f"BIC Value: {str(round(gaussian_glm.bic_llf, 5))}")
        

    log.info (f"{filename} saved to .../results/glm/{index.upper()} directory")

# --------------------------------------------------------------------------------------------------------------------

# codespace starts HERE
# for each_index in ['msci', 'nasdaq', 'ftse', 'stoxx', 'snp']:
#     gaussian_glm(
#         index=each_index,
#         measure='roe',
#         scores=True
#     )

#     gaussian_glm(
#         index=each_index,
#         measure='roa',
#         scores=True
#     )

    # gaussian_glm(
    #     index=each_index,
    #     measure='roe',
    #     scores=False
    # )

    # gaussian_glm(
    #     index=each_index,
    #     measure='roa',
    #     scores=False
    # )

# TODO: Correlation, Relationship, Regression, Cluster Analysis 
#       (cross-compare with methods on cited literatures on types of analysis used)
