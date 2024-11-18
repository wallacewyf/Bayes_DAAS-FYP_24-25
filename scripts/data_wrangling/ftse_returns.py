# import parameter file
import config

# import libraries
import pandas as pd

year_list = [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

# read returns data file
returns_df = pd.read_excel(config.ftse_returns, index_col=[0,1], skiprows=2)
returns_df.index.names = ['Identifier', 'Company Name']

# Return on Equity  - Actual
roe_df = returns_df.iloc[:, :20]
roe_df = roe_df.set_axis(year_list, axis=1)
roe_df.dropna(inplace=True)

# Return on Assets - Actual
roa_df = returns_df.iloc[:, 20:40]
roa_df = roa_df.set_axis(year_list, axis=1)
roa_df.dropna(inplace=True)

# 52-Week Total Return
yoy_return = returns_df.iloc[2:, 40:60]
yoy_return = yoy_return.set_axis(year_list, axis=1)
yoy_return.dropna(inplace=True)

# Market Capitalisation
mkt_cap = returns_df.iloc[2:, 60:80]
mkt_cap = mkt_cap.set_axis(year_list, axis=1)
mkt_cap.dropna(inplace=True)

# Total Assets - Reported
ta_df = returns_df.iloc[2:, 80:100]
ta_df = ta_df.set_axis(year_list, axis=1)
ta_df.dropna(inplace=True)

# Q Ratio
q_mktcap = mkt_cap.apply(pd.to_numeric, errors='coerce')    # convert to numeric
q_ta = ta_df.apply(pd.to_numeric, errors='coerce')          # convert to numeric

q_ta, q_mktcap = q_ta.align(q_mktcap, join='inner')         # inner join
q_ratio = q_mktcap / q_ta

# CHECK IF INNER JOIN SUCCESS
# print(q_mktcap.index.equals(q_ta.index))  # Should return True
# print(q_mktcap.columns.equals(q_ta.columns))  # Should return True

q_ratio.to_csv(config.results_path + 'q_ratio.csv', header=True, index=True)        # exports to csv format

overval_2023 = q_ratio[q_ratio[2023] > 1][2023]
underval_2023 = q_ratio[(q_ratio[2023] > 0) & (q_ratio[2023] <= 0.05)][2023]
fairval_2023 = q_ratio[(q_ratio[2023] > 0.05) & (q_ratio[2023] <= 1)][2023]

for each_company in underval_2023.index.values:
    print (list(each_company)[1])
