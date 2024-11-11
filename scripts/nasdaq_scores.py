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

print (g_pillars)

# # clean up and set index
# source_df['Identifier'] = source_df['Identifier'].apply(lambda x: x[:len(x)-3])
# source_df.set_index(keys='Identifier', inplace=True)

# # data wrangling
# source_df.drop('Exchange Ticker', 
#                axis=1, 
#                inplace=True)

# source_df.rename(columns={'Company Market Capitalization\n(USD)':'Market Cap (USD $)',
#                           'ESG Score\n(FY0)': 'ESG Score',
#                           'ESG Score Grade\n(FY0)': 'ESG Grade', 
#                           'Environmental Pillar Score Grade\n(FY0)': 'E Grade',
#                           'Social Pillar Score Grade\n(FY0)': 'S Grade',
#                           'Governance Pillar Score Grade\n(FY0)': 'G Grade',
#                           'Environmental Pillar Score\n(FY0)': 'E Score', 
#                           'Social Pillar Score\n(FY0)': 'S Score',
#                           'Governance Pillar Score\n(FY0)': 'G Score',
#                           '52 Week Total Return': '52 Week Total Return (%)'},
#                   inplace=True)

# # convert YoY into %
# source_df['52 Week Total Return (%)'] = source_df['52 Week Total Return (%)'].apply(lambda x: round(x*100, 2))

# # sort ESG scores and export
# source_df.sort_values(by=['ESG Score', 'Company Name'], 
#                       ascending=[False, True], 
#                       axis=0, 
#                       inplace=True)

# source_df.to_csv(config.results_path + config.combined_esg_filename, 
#                  sep = ',', 
#                  header=True, 
#                  index=False)

# print ('ESG scores are now sorted from highest to lowest.',
#        '\n' + f'File has been saved to {config.results_path}' + f'{config.combined_esg_filename}.' + '\n')

# print ('------------------------------------------------------------')

# # scatterplot - ESG combined scores v. 52W total return
# x = source_df['ESG Score'].tolist()
# y = source_df['52 Week Total Return (%)'].tolist()

# slope, intercept, r, p, std_err = stats.linregress(x, y)


# def myfunc(x):
#   return slope * x + intercept

# mymodel = list(map(myfunc, x))

# print ('x: Combined ESG scores')
# print ('y: FY0 returns' + '\n')
# print ('The relationship between x and y is:', round(r, 5))
# print ('------------------------------------------------------------')

# print (source_df.head(5))