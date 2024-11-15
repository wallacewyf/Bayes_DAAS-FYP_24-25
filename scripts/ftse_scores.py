# import parameters file
import config

# import libraries
import pandas as pd

# init source df
source_df = pd.read_excel(config.ftse_esg, index_col=[0,1], skiprows=1)
source_df.index.names = ['Identifier', 'Company Name']

column_names = [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004]

esg_scores = source_df.iloc[:, :20]
esg_scores = esg_scores.set_axis(column_names, axis=1)
esg_scores.dropna(inplace=True)
esg_scores.sort_values(by=['Company Name', 2023],
                       axis=0,
                       ascending=[True, False],
                       inplace=True)

e_pillars = source_df.iloc[:, 20:40]
e_pillars = e_pillars.set_axis(column_names, axis=1)
e_pillars.dropna(inplace=True)
e_pillars.sort_values(by=['Company Name', 2023],
                       axis=0,
                       ascending=[True, False],
                       inplace=True)

s_pillars = source_df.iloc[:, 40:60] 
s_pillars = s_pillars.set_axis(column_names, axis=1)
s_pillars.dropna(inplace=True)
s_pillars.sort_values(by=['Company Name', 2023],
                       axis=0,
                       ascending=[True, False],
                       inplace=True)

g_pillars = source_df.iloc[:, 60:80]
g_pillars = g_pillars.set_axis(column_names, axis=1)
g_pillars.dropna(inplace=True)
g_pillars.sort_values(by=['Company Name', 2023],
                       axis=0,
                       ascending=[True, False],
                       inplace=True)