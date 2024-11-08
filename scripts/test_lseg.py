import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt

source_path = '/Users/wallace/Documents/code/python/fyp_24/'
source_file = 'FY2023_ESG_Returns.xlsx'

source = source_path + source_file
combined_esg_filename = 'sorted_combined_esg.csv'

source_df = pd.read_excel(source, header=0)

market_cap = source_df[['Exchange Ticker', 
                        'Company Name', 
                        'Company Market Capitalization\n(USD)'
]]

combined_esg = source_df[['Exchange Ticker', 
                        'Company Name', 
                        'ESG Score\n(FY0)',
                        'ESG Score Grade\n(FY0)']]

e_score = source_df[['Exchange Ticker', 
                    'Company Name', 
                    'Environmental Pillar Score Grade\n(FY0)',
                     'Environmental Pillar Score\n(FY0)']]

s_score = source_df[['Exchange Ticker', 
                    'Company Name', 
                    'Social Pillar Score Grade\n(FY0)',
                     'Social Pillar Score\n(FY0)']]

g_score = source_df[['Exchange Ticker', 
                    'Company Name', 
                    'Governance Pillar Score Grade\n(FY0)',
                     'Governance Pillar Score\n(FY0)']]

yoy_return = source_df[['Exchange Ticker', 
                        'Company Name', 
                        '52 Week Total Return']]

# convert into %
yoy_return['52 Week Total Return'] = yoy_return['52 Week Total Return'].apply(lambda x: round(x*100, 2))
yoy_return.rename(columns={'52 Week Total Return':'52 Week Total Return (%)'}, inplace=True)

# sort ESG scores
combined_esg.sort_values(by='ESG Score\n(FY0)', ascending=False, axis=0, inplace=True)
combined_esg.to_csv(source_path + combined_esg_filename, sep = ',', header=True, index=False)

print ('ESG scores are now sorted from highest to lowest.',
       '\n' + f'File has been saved to {source_path}' + f'{combined_esg_filename}.')

# scatterplot - ESG combined scores v. 52W total return
x = combined_esg['ESG Score\n(FY0)'].tolist()
y = yoy_return['52 Week Total Return (%)'].tolist()

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

print (r)

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
