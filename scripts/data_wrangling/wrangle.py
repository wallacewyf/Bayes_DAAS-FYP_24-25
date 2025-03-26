# import parameter file
import config

# import libraries
import pandas as pd

# initialize logger module 
log = config.logging

returns_col  = list(range(2004,2024)) * 6
scores_col = list(range(2004, 2024)) * 4
index_names = ['Company Name', 'GICS Sector Name']

# According to GICS Classification - only Financial and Technology companies
# Official classification names can be found here: 
# MSCI, 2025. The Global Industry Classification Standard (GICS®) 
# Available at: https://www.msci.com/our-solutions/indexes/gics (Accessed: 13 March 2025)

sector_scope = ['Financials', 'Information Technology']

# Financial Returns
# ===========================================================================
log.info (f"Reading Financial Returns data...")
returns = pd.read_excel(config.returns,
                        index_col = [1,2],
                        skiprows=2)

returns.index.names = index_names

# drop Identifier (Company RIC Code).
returns = returns.iloc[:, 1:]

# sort by ascending order on Company Name
returns.sort_values(by=['Company Name'],
                   axis=0, 
                   ascending=True, 
                   inplace=True)

# set column headers to be 2004, 2005, ..., 2023
returns = returns.set_axis(returns_col, axis=1)

log.warning(f"Melting dataframes into long format...")

# Return on Equity - Actual
roe = returns.iloc[:, :20].melt(var_name='Year', 
                                value_name='ROE',
                                ignore_index=False)

# Return on Assets - Actual 
roa = returns.iloc[:, 20:40].melt(var_name='Year', 
                                value_name='ROA',
                                ignore_index=False).drop(columns=['Year'])

# Net Income - Actual
net_income = returns.iloc[:, 80:100].melt(var_name='Year', 
                                value_name='Net_Income',
                                ignore_index=False).drop(columns=['Year'])

# Shareholders' Equity
eqi = returns.iloc[:, 100:].melt(var_name='Year', 
                                value_name='EQI',
                                ignore_index=False).drop(columns=['Year'])

# Company Market Cap
mktcap = returns.iloc[:, 40:60]

# Total Assets, Reported
ta = returns.iloc[:, 60:80]

log.info (f"Computing Q Ratio Calculations...")

q = mktcap / ta
q = q.melt(var_name='Year', 
            value_name='Q_Ratio',
            ignore_index=False).drop(columns=['Year'])

mktcap = mktcap.melt(var_name='Year', 
                    value_name='MKTCAP',
                    ignore_index=False).drop(columns=['Year'])

ta = ta.melt(var_name='Year', 
            value_name='TA',
            ignore_index=False).drop(columns=['Year'])

log.info (f"Merging all melted dataframes into a single dataframe...")
returns = pd.concat([roe, roa, mktcap, ta, q, net_income, eqi],
                    axis=1)

returns.set_index('Year', 
                  append=True, 
                  inplace=True)

log.info (f"Dropping duplicate rows in returns dataframe...")
returns.drop_duplicates(inplace=True)

log.warning (f"Running filter 1: ROE is NA but Net_Income and EQI not NA values")
# Condition 1: ROE is NA but Net_Income and EQI not NA
cond1 = returns['ROE'].isna() & returns['Net_Income'].notna() & returns['EQI'].notna()

log.info (f"Calculating replacement ROE value...")
returns.loc[cond1, 'ROE'] = returns.loc[cond1, 'Net_Income'] / returns.loc[cond1, 'EQI']

log.warning (f"Running filter 2: ROA is NA but Net_Income and TA not NA")
# Condition 2: ROA is NA but Net_Income and TA not NA
cond2 = returns['ROA'].isna() & returns['Net_Income'].notna() & returns['TA'].notna()

log.info (f"Calculating replacement ROA value...")
returns.loc[cond2, 'ROA'] = returns.loc[cond2, 'Net_Income'] / returns.loc[cond2, 'TA']

# Now omit MKTCAP, TA, Net_Income and EQI since will not be used in regression model
returns = returns.iloc[:, [0,1,4]]

# ESG Scores 
# ===========================================================================
log.info (f"Reading ESG data...")
scores = pd.read_excel(config.scores,
                       index_col = [1,2],
                       skiprows = 2)

scores.index.names = index_names

# drop Identifier (Company RIC Code)
scores = scores.iloc[:, 1:]

# set column headers to be 2004, 2005, ..., 2023
scores = scores.set_axis(scores_col, axis=1)

# sorting in ascending order by Company Name
scores.sort_values(by=['Company Name'],
                   axis=0, 
                   ascending=True, 
                   inplace=True)

log.warning (f"Dropping duplicated rows in ESG scores dataframe...")
scores.drop_duplicates(inplace=True)

# ESG Scores
esg = scores.iloc[:, :20]

# E, S, G Individual Pillars
log.info (f"Melting ESG dataframe into long format...")
e_df = scores.iloc[:, 20:40]
s_df = scores.iloc[:, 40:60]
g_df = scores.iloc[:, 60:]

esg = esg.melt(var_name='Year', 
                value_name='ESG',
                ignore_index=False)

e_df = e_df.melt(var_name='Year', 
                value_name='E',
                ignore_index=False).drop(columns=['Year'])

s_df = s_df.melt(var_name='Year', 
                value_name='S',
                ignore_index=False).drop(columns=['Year'])

g_df = g_df.melt(var_name='Year', 
                value_name='G',
                ignore_index=False).drop(columns=['Year'])

log.info (f"Merging melted arrays into scores dataframe... ")
scores = pd.concat([esg, e_df, s_df, g_df], 
                   axis=1)

scores.set_index('Year', append=True, inplace=True)

log.warning (f"Running filter 3: ROE, ROA, ESG are not NAs\n")

# Condition 3: ROE, ROA, ESG are not NAs
returns, scores = returns.align(scores, join='inner', axis=0)
cond3 = (returns['ROE'].notna() 
         & returns['ROA'].notna() 
         & returns['Q_Ratio'].notna() 
         & scores['ESG'].notna())

scores = scores[cond3]
returns = returns[cond3]

# Combine both Returns and Scores into 1 single dataframe 
df = pd.concat([scores, returns], axis=1)

# Remove outliers from ROE / ROA since ESG is fixed
# Top 5% outliers - subject to change

top_10_ROE = df['ROE'].quantile(0.95)
bot_10_ROE = df['ROE'].quantile(0.05)

top_10_ROA = df['ROA'].quantile(0.95)
bot_10_ROA = df['ROA'].quantile(0.05)

df = df[
    (df['ROA'] >= bot_10_ROA) & (df['ROA'] <= top_10_ROA) &
    (df['ROE'] >= bot_10_ROE) & (df['ROE'] <= top_10_ROE)
]

# Convert ROE / ROA from 0 - 1 to 0 - 100%
df['ROE'] *= 100
df['ROA'] *= 100

df.to_csv(config.data_path + 'cleaned_data.csv')

# available dataframes
# GICS Sectors
    # Financials
    # Utilities
    # Health Care
    # Information Technology
    # Materials
    # Industrials
    # Energy
    # Consumer Staples
    # Consumer Discretionary
    # Real Estate
    # Communication Services

finance = df[df.index.get_level_values(1) == 'Financials']
utils = df[df.index.get_level_values(1) == 'Utilities']
health = df[df.index.get_level_values(1) == 'Health Care']
tech = df[df.index.get_level_values(1) == 'Information Technology']
materials = df[df.index.get_level_values(1) == 'Materials']
industrials = df[df.index.get_level_values(1) == 'Industrials']
energy = df[df.index.get_level_values(1) == 'Energy']
consumer_staple = df[df.index.get_level_values(1) == 'Consumer Staples']
consumer_disc = df[df.index.get_level_values(1) == 'Consumer Discretionary']
reits = df[df.index.get_level_values(1) == 'Real Estate']
comm = df[df.index.get_level_values(1) == 'Communication Services']

log.info (f"Data wrangling complete!\n")

# Debugger
# ===========================================================================


def notes():
    '''
    To do the threshold cutoff after Kyoto/Paris to see if impact is more prominent
    Since impact might change according to regulation enforcement time

    Lag should not be included since time period has been shortened
    i.e. analysis would be more on over a n-year period the impact is upwards/downwards

    Finance - linear regression
        - threshold regression to see if it performs better than overall regression

    Tech - not linear regression / to work on GLM 
        - if GLM doesn't work then try threshold regression 


    Note: to check if linear regression works (i.e. linearity between explanatory and predictor variable)
        - residuals of a linear regression should be normally distributed


    Expected: 
        finance - yes, impact on returns from ESG scores

        tech - maybe not, since technology companies performance can be weird

    Note: it's also possible that if linear regression work with ROE, it's possible that
    it'll be GLM for ROA as well (applicable to both finance/tech)

    ASOF 20-Mar:
    Finance:
        - ln(ROE): linear regression OK
        - ROA: linear regression OK

    Technology:
        - ROE: doesn't work linear reg/GLM
        - ROA: haven't tried yet

    ASOF 23-Mar:
    Finance:
        - All OK - models to be verified by David
        
        Concluded: not OK - QQ-plot is weird

    Discussion with David:
        - the norm in financial data is to log (but probably needs academic reference)
        - remove top 5/10% of outliers to remove noise from the model 
            - but has to be specified in methodology since we're only assessing
            - the impact of the framework on returns

        Code:
            - multiple functions to try different models 
            - since reader might try different function 
            - basically there should be a function for each type of model i.e. gamma/gaussian glm, linear reg, linear reg w log
            - brief explanation on methodology section in paper
            - refer to previous functions in regression_OLD.py

        Paper: 
            write about the trial and error process, but brief description on the error and a more detailed 
            one for the best fit one (evaluate according to p-value)


    '''