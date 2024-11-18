# import parameter file
import config

# import libraries
import pandas as pd

col_names = ['GICS Industry Name', 'Exchange Name', 
             2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

# read returns data file
returns_df = pd.read_excel(config.ftse_returns, index_col=[0,1], skiprows=2)
returns_df.index.names = ['Identifier', 'Company Name']

# Return on Equity  - Actual
roe_df = returns_df.iloc[:, :22]
roe_df = roe_df.set_axis(col_names, axis=1)
roe_df.dropna(inplace=True)

# Return on Assets - Actual
roa_df = returns_df.iloc[:, [0,1] + list(range(22,42))]
roa_df = roa_df.set_axis(col_names, axis=1)
roa_df.dropna(inplace=True)

# 52-Week Total Return
yoy_return = returns_df.iloc[2:, [0,1] + list(range(42,62))]
yoy_return = yoy_return.set_axis(col_names, axis=1)
yoy_return.dropna(inplace=True)

# Market Capitalisation
mkt_cap = returns_df.iloc[2:, [0,1] + list(range(62,82))]
mkt_cap = mkt_cap.set_axis(col_names, axis=1)
mkt_cap.dropna(inplace=True)

# Total Assets - Reported
ta_df = returns_df.iloc[2:, [0,1] + list(range(82,102))]
ta_df = ta_df.set_axis(col_names, axis=1)
ta_df.dropna(inplace=True)

# Total Liabilities
tl_df = returns_df.iloc[2:, [0,1] + list(range(102,122))]
tl_df = tl_df.set_axis(col_names, axis=1)
tl_df.dropna(inplace=True)

# Total Equity
te_df = returns_df.iloc[2:, [0,1] + list(range(122,142))]
te_df = te_df.set_axis(col_names, axis=1)
te_df.dropna(inplace=True)

# Minority Interest
minority_df = returns_df.iloc[2:, [0,1] + list(range(142,162))]
minority_df = minority_df.set_axis(col_names, axis=1)
minority_df.dropna(inplace=True)

# P/E Ratio
pe_df = returns_df.iloc[2:, [0,1] + list(range(162,182))]
pe_df = minority_df.set_axis(col_names, axis=1)
pe_df.dropna(inplace=True)

# Q Ratio
q_mktcap = mkt_cap.apply(pd.to_numeric, errors='coerce')    # convert to numeric
q_ta = ta_df.apply(pd.to_numeric, errors='coerce')          # convert to numeric

q_ta, q_mktcap = q_ta.align(q_mktcap, join='inner')         # inner join
q_ratio = q_mktcap / q_ta

# Q Ratio test

q_mktcap2 = mkt_cap.apply(pd.to_numeric, errors='coerce')
q_ta2 = ta_df.apply(pd.to_numeric, errors='coerce')
q_tl = tl_df.apply(pd.to_numeric, errors='coerce')
q_te = te_df.apply(pd.to_numeric, errors='coerce')
q_minority = minority_df.apply(pd.to_numeric, errors='coerce')

q_df = pd.concat([q_mktcap2, 
                  q_ta2,
                  q_tl, 
                  q_te, 
                  q_minority],
                  join='inner',
                  axis=1)

q_df.drop(['GICS Industry Name', 'Exchange Name'], axis=1, inplace=True)

q_mktcap2 = q_df.iloc[:, :20]
q_ta2 = q_df.iloc[:, 20:40]
q_tl = q_df.iloc[:, 40:60]
q_te = q_df.iloc[:, 60:80]
q_minority= q_df.iloc[:, 80:100]

q_ratio = (q_mktcap2 + q_tl + q_minority) / q_ta2

q_ratio.to_csv(config.results_path + 'q_ratio.csv', header=True, index=True)

# CHECK IF INNER JOIN SUCCESS
# print(q_mktcap.index.equals(q_ta.index))  # Should return True
# print(q_mktcap.columns.equals(q_ta.columns))  # Should return True

# q_ratio.to_csv(config.results_path + 'q_ratio.csv', header=True, index=True)        # exports to csv format

# overval_2023 = q_ratio[q_ratio[2023] > 1][2023]
# underval_2023 = q_ratio[(q_ratio[2023] > 0) & (q_ratio[2023] <= 0.05)][2023]
# fairval_2023 = q_ratio[(q_ratio[2023] > 0.05) & (q_ratio[2023] <= 1)][2023]

# for each_company in underval_2023.index.values:
#     print (list(each_company)[1])
