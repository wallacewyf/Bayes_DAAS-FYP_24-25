# import parameter file
import config

# import libraries
import pandas as pd

year_list = [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

# read nasdaq return file
returns_df = pd.read_excel(config.nasdaq_returns, index_col=[0,1])

# Return on Equity  - Actual
roe_df = returns_df.iloc[2:, :20]
roe_df = roe_df.set_axis(year_list, axis=1)
roe_df.dropna(inplace=True)

# Return on Assets - Actual
roa_df = returns_df.iloc[2:, 20:40]
roa_df = roa_df.set_axis(year_list, axis=1)
roa_df.dropna(inplace=True)

# 52-Week Total Return
yoy_return = returns_df.iloc[2:, 80:100]
yoy_return = yoy_return.set_axis(year_list, axis=1)
yoy_return.dropna(inplace=True)

# Market Capitalisation
mkt_cap = returns_df.iloc[2:, 100:120]
mkt_cap = mkt_cap.set_axis(year_list, axis=1)
mkt_cap.dropna(inplace=True)

# Total Assets - Reported
ta_df = returns_df.iloc[2:, 120:140]
ta_df = ta_df.set_axis(year_list, axis=1)
ta_df.dropna(inplace=True)

# P/E Ratio
pe_ratio = returns_df.iloc[2:, 140:160]
pe_ratio = pe_ratio.set_axis(year_list, axis=1)
pe_ratio.dropna(inplace=True)
