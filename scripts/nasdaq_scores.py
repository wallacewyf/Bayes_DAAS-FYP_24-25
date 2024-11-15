# import parameters file
import config

# import libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt

# init source df
source_df = pd.read_excel(config.nasdaq_esg, index_col=0, skiprows=2)

column_names = ['Company Name','GICS Industry Name',
                2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

esg_scores = source_df.iloc[:, [0,1] + list(range(23,43))]
esg_scores = esg_scores.set_axis(column_names, axis=1)
esg_scores.dropna(inplace=True)
esg_scores.sort_values(by=[2023, 'GICS Industry Name', 'Company Name'],
                       axis=0,
                       ascending=[False, True, True],
                       inplace=True)

e_pillars = source_df.iloc[:, [0,1] + list(range(43,63))]
e_pillars = e_pillars.set_axis(column_names, axis=1)
e_pillars.dropna(inplace=True)
e_pillars.sort_values(by=[2023, 'GICS Industry Name', 'Company Name'],
                       axis=0,
                       ascending=[False, True, True],
                       inplace=True)

s_pillars = source_df.iloc[:, [0,1] + list(range(63,83))] 
s_pillars = s_pillars.set_axis(column_names, axis=1)
s_pillars.dropna(inplace=True)
s_pillars.sort_values(by=[2023, 'GICS Industry Name', 'Company Name'],
                       axis=0,
                       ascending=[False, True, True],
                       inplace=True)

g_pillars = source_df.iloc[:, [0,1] + list(range(83,103))]
g_pillars = g_pillars.set_axis(column_names, axis=1)
g_pillars.dropna(inplace=True)
g_pillars.sort_values(by=[2023, 'GICS Industry Name', 'Company Name'],
                       axis=0,
                       ascending=[False, True, True],
                       inplace=True)