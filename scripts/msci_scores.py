# import libraries
import pandas as pd

# import parameters file
import config

# init source df
source_df = pd.read_excel(config.msci_esg, index_col=0, skiprows=1)

column_names = ['Company Name','GICS Industry Name', 'Exchange Name',
                2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

esg_scores = source_df.iloc[:, :23]
esg_scores = esg_scores.set_axis(column_names, axis=1)
esg_scores.dropna(inplace=True)
esg_scores.sort_values(by=['Company Name', 'GICS Industry Name', 2023],
                       axis=0,
                       ascending=[True, True, False],
                       inplace=True)

e_pillars = source_df.iloc[:, [0,1,2] + list(range(23,43))]
e_pillars = e_pillars.set_axis(column_names, axis=1)
e_pillars.dropna(inplace=True)
e_pillars.sort_values(by=['Company Name', 'GICS Industry Name', 2023],
                       axis=0,
                       ascending=[True, True, False],
                       inplace=True)

s_pillars = source_df.iloc[:, [0,1,2] + list(range(43,63))] 
s_pillars = s_pillars.set_axis(column_names, axis=1)
s_pillars.dropna(inplace=True)
s_pillars.sort_values(by=['Company Name', 'GICS Industry Name', 2023],
                       axis=0,
                       ascending=[True, True, False],
                       inplace=True)

g_pillars = source_df.iloc[:, [0,1,2] + list(range(63,83))]
g_pillars = g_pillars.set_axis(column_names, axis=1)
g_pillars.dropna(inplace=True)
g_pillars.sort_values(by=['Company Name', 'GICS Industry Name', 2023],
                       axis=0,
                       ascending=[True, True, False],
                       inplace=True)