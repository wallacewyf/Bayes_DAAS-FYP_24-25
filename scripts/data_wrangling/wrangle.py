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

# Shareholders' Equity - Actual
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
returns = pd.concat([roa, roe, mktcap, ta, q, net_income, eqi],
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
df = pd.concat([returns, scores], axis=1)

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

# Convert ROE / ROA from 0 - 1 to 0 - 100% scale
df['ROE'] *= 100
df['ROA'] *= 100

# sorting in ascending order by Company Name
df.sort_values(by=['Year', "Company Name"],
                   axis=0, 
                   ascending=[True, True],
                   inplace=True)

scores.to_csv(config.data_path + 'scores.csv')
returns.to_csv(config.data_path + 'returns.csv')

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
all = df.iloc[:,:-1]

log.info (f"Data wrangling complete!\n")

# Debugger
# ===========================================================================
