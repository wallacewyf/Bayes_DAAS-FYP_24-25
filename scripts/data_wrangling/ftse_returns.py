# import parameter file
import config

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

col_names = list(range(2023, 2003, -1))

# read returns data file
returns_df = pd.read_excel(config.ftse_returns, index_col=[0,1,2], skiprows=2)
returns_df.index.names = ['Identifier', 'Company Name', 'GICS Industry Name']
returns_df.drop(returns_df.columns[0], axis=1, inplace=True)

# Return on Equity  - Actual
roe_df = returns_df.iloc[:, :20]
roe_df = roe_df.set_axis(col_names, axis=1)
roe_df.dropna(inplace=True)

# Return on Assets - Actual
roa_df = returns_df.iloc[:, 20:40]
roa_df = roa_df.set_axis(col_names, axis=1)
roa_df.dropna(inplace=True)

# 52-Week Total Return
yoy_return = returns_df.iloc[:, 40:60]
yoy_return = yoy_return.set_axis(col_names, axis=1)
yoy_return.dropna(inplace=True)

# Market Capitalisation
mkt_cap = returns_df.iloc[:, 60:80]
mkt_cap = mkt_cap.set_axis(col_names, axis=1)
mkt_cap.dropna(inplace=True)

# Total Assets - Reported
ta_df = returns_df.iloc[:, 80:100]
ta_df = ta_df.set_axis(col_names, axis=1)
ta_df.dropna(inplace=True)

# Total Liabilities
tl_df = returns_df.iloc[:, 100:120]
tl_df = tl_df.set_axis(col_names, axis=1)
tl_df.dropna(inplace=True)

# Total Equity
te_df = returns_df.iloc[:, 120:140]
te_df = te_df.set_axis(col_names, axis=1)
te_df.dropna(inplace=True)

# Minority Interest
minority_df = returns_df.iloc[:, 140:160]
minority_df = minority_df.set_axis(col_names, axis=1)
minority_df.dropna(inplace=True)

# P/E Ratio
pe_df = returns_df.iloc[:, 160:180]
pe_df = minority_df.set_axis(col_names, axis=1)
pe_df.dropna(inplace=True)

# Q Ratio
q_ta, q_mktcap = ta_df.align(mkt_cap, join='inner')
q_ratio = q_mktcap / q_ta
q_ratio.insert(0, column='Exchange Name', value='LONDON STOCK EXCHANGE')
q_ratio.set_index('Exchange Name', append=True, inplace=True)

# CHECK IF INNER JOIN SUCCESS
# print(q_mktcap.index.equals(q_ta.index))  # Should return True
# print(q_mktcap.columns.equals(q_ta.columns))  # Should return True