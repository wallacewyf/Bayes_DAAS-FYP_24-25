# import parameter file
import config

# import libraries
import eikon as ek
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt

yeardict_map = {'Return On Equity - Actual\nIn the last 20 FY': '2023',
                       'Unnamed: 3': '2022',
                        'Unnamed: 4': '2021',
                        'Unnamed: 5': '2020',
                        'Unnamed: 6': '2019',
                        'Unnamed: 7': '2018',
                        'Unnamed: 8': '2017',
                        'Unnamed: 9': '2016',
                        'Unnamed: 10': '2015',
                        'Unnamed: 11': '2014',
                        'Unnamed: 12': '2013',
                        'Unnamed: 13': '2012',
                        'Unnamed: 14': '2011',
                        'Unnamed: 15': '2010',
                        'Unnamed: 16': '2009',
                        'Unnamed: 17': '2008',
                        'Unnamed: 18': '2007',
                        'Unnamed: 19': '2006',
                        'Unnamed: 20': '2005',
                        'Unnamed: 21': '2004'}

year_list = ['Company Name',2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

# read nasdaq return file
returns_df = pd.read_excel(config.nasdaq_returns, index_col=0)

# Return on Equity  - Actual
roe_df = returns_df.iloc[2:, :21]
roe_df.rename(columns=yeardict_map, inplace=True)
roe_df.dropna(inplace=True)

# Return on Assets - Actual
roa_df = returns_df.iloc[2:, [0] + list(range(21, 41))]
roa_df = roa_df.set_axis(year_list, axis=1)
roa_df.dropna(inplace=True)

# 52-Week Total Return
yoy_return = returns_df.iloc[2:, [0] + list(range(81, 101))]
yoy_return = yoy_return.set_axis(year_list, axis=1)
yoy_return.dropna(inplace=True)

# Market Capitalisation
mkt_cap = returns_df.iloc[2:, [0] + list(range(101, 121))]
mkt_cap = mkt_cap.set_axis(year_list, axis=1)
mkt_cap.dropna(inplace=True)

# Total Assets - Reported
ta_df = returns_df.iloc[2:, [0] + list(range(121, 141))]
ta_df = ta_df.set_axis(year_list, axis=1)
ta_df.dropna(inplace=True)

# P/E Ratio
pe_ratio = returns_df.iloc[2:, [0] + list(range(141, 161))]
pe_ratio = pe_ratio.set_axis(year_list, axis=1)
pe_ratio.dropna(inplace=True)