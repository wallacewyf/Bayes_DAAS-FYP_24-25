# packages
import sys, os, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# statistical packages
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score                        # R-squared
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

import statsmodels.api as sm

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

try:
    logging.basicConfig(
                        filename=config.log + "/log.txt",
                        format='[%(asctime)s] [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a',
                        level=logging.DEBUG
                        )

except FileNotFoundError:
    os.mkdir(config.log)

# split between pre/post COVID pandemic / Paris Agreement
# print (esg_sum.loc[:, [2023, 2022, 2021, 2020, 2019, 2017, 2016, 2015, 2014, 2013, 2008, 2004]])

# --------------------------------------------------------------------
# Regression Analysis 
# notes: 
# After discussion with Pietro, run a normal linear regression first by ensuring all returns 
# are normally distributed. 
# If normally distributed, linearity is proven and therefore linear or multivariate regression 
# could then be fitted.

# Check if normally distributed.

roe_df = wrangle.returns('msci', 'roe')
roe_df.reset_index(inplace=True)

roe_df = roe_df.melt(
    id_vars=['Identifier', 'Company Name', 'GICS Industry Name', 'Exchange Name'],
    var_name='Year',
    value_name='ROE'
)

roe_df.dropna(inplace=True)

roe_data = roe_df['ROE']

print (roe_df.query('ROE < -100'))

print (roe_data.describe())

# 1. Histogram
# plt.figure(figsize=(10, 4))
# sns.histplot(roe_data, kde=True, bins=30)
# plt.title('Histogram of ROE')
# plt.xlabel('ROE')
# plt.ylabel('Frequency')
# plt.show()

# 2. Q-Q Plot
# plt.figure(figsize=(6, 6))
# sm.qqplot(roe_data, line='45', fit=True)
# plt.title('Q-Q Plot of ROE')
# plt.show()

# --------------------------------------------------------------------
# Correlation Analysis

# TODO: return all correlation in a suitable dataframe for export to .csv for further analysis

# Correlation between Index Components and Financial Returns
def index_corr(df_1, df_2):
    index_corr_arr = []
    logging.info ('Running correlation analysis between index components and financial returns...')
    logging.info ('--------------------------------------------------------------------')

    while sum(df_1.isnull().sum()) != sum(df_2.isnull().sum()):

        # TODO: dropna(axis = 1, inplace = True)
        df_1.dropna(inplace=True)
        df_2.dropna(inplace=True)

        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            logging.info (f"{len(df_1.columns)} columns successfully aligned and extracted")
            total_year_avg = 0

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                logging.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum
            
            index_corr_arr.append(total_year_avg/len(df_1.columns))

            break
        
        logging.info ('')
        logging.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        logging.info ('--------------------------------------------------------------------')

        break

    else:
        logging.info ('Running correlation analysis...')
        logging.info ('--------------------------------------------------------------------')

        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            logging.info (f"{len(df_1.columns)} columns successfully aligned and extracted")

            total_year_avg = 0

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                logging.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum

            index_corr_arr.append(total_year_avg/len(df_1.columns))

            break
        
        logging.info ('')
        logging.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        logging.info ('--------------------------------------------------------------------')

    return index_corr_arr

# Correlation between GICS Industry Name and Financial Returns
def industry_corr(df_1, df_2):
    industry_corr_arr = []

    logging.info ('Running correlation analysis between industries and financial returns...')
    logging.info ('--------------------------------------------------------------------')

    while sum(df_1.isnull().sum()) != sum(df_2.isnull().sum()):
        df_1.dropna(inplace=True)
        df_2.dropna(inplace=True)

        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            logging.info (f"{len(df_1.columns)} columns successfully aligned and extracted")

            df_1 = df_1.groupby(['GICS Industry Name']).mean()      # ESG Scores
            df_2 = df_2.groupby(['GICS Industry Name']).mean()      # Financial Ratio

            total_year_avg = 0

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                logging.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum

            industry_corr_arr.append(total_year_avg/len(df_1.columns))

            break
        
        logging.info ('')
        logging.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        logging.info ('--------------------------------------------------------------------')

        break

    else:
        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            logging.info (f"{len(df_1.columns)} columns successfully aligned and extracted")

            total_year_avg = 0

            df_1 = df_1.groupby(['GICS Industry Name']).mean()      # ESG Scores
            df_2 = df_2.groupby(['GICS Industry Name']).mean()      # Financial Ratio

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                logging.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum
            
            industry_corr_arr.append(total_year_avg/len(df_1.columns))

            break

        logging.info ('')
        logging.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        logging.info ('--------------------------------------------------------------------')

    print (df_1, df_2)

    return industry_corr_arr

index = ['nasdaq', 'snp', 'stoxx', 'ftse', 'msci']
measures = ['roe', 'roa', 'mktcap', 'q']
esg = ['esg', 'e', 's', 'g']

def init_corr():
    print ('Initializing correlation analysis...')

    for each_index in index:
        for each_measure in measures:
            for each_esg in esg:
                logging.info (f'Correlation by index initializing...')

                index_val = index_corr(df_1 = wrangle.scores(each_index, each_esg), 
                                    df_2 = wrangle.returns(each_index, each_measure))
                
                logging.info (f'Correlation by index completed.')
                logging.info (f'Correlation by industry initializing...')

                industry_val = industry_corr(df_1= wrangle.scores(each_index, each_esg),
                                        df_2 = wrangle.returns(each_index, each_measure))
                
                logging.info (f"Correlation by industry completed.")
                logging.info (f"Correlation analysis between {each_index.upper()}'s {each_esg.upper()} scores and {each_measure.upper()} completed.")

    print ('Correlation Analysis completed.')
    print ('Check log file for results.')

    return [index_val, industry_val]


# industry_corr(
#     df_1 = wrangle.scores('nasdaq', 'esg'),
#     df_2 = wrangle.returns('nasdaq', 'roe')
# )

# Data Visualization
# line graph for relationship between average of Q ratio and ESG scores
# esg_avg = esg_scores.mean(axis=0)
# q_avg = q_ratio.mean(axis=0)

# years = np.arange(2023, 2003, -1)
# figure, ax1 = plt.subplots(figsize=(10, 6))

# # primary axis
# ax1.plot(years, esg_avg, label='Average ESG Scores', color='blue')
# ax1.set_xlabel('Year')
# ax1.set_ylabel('Average ESG Scores', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.set_xticks(np.arange(2024, 2002, -2))

# # secondary axis
# ax2 = ax1.twinx()
# ax2.plot(years, q_avg, label="Average Tobin's Q", color='green')
# ax2.set_ylabel("Average Tobin's Q", color='green')
# ax2.tick_params(axis='y', labelcolor='green')

# # title, legend, label
# plt.title('Yearly Trends of ESG Scores and Tobin\'s Q')
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

# plt.show()

# print (q_ratio.columns)

# q_ratio_x = q_ratio.columns
# q_ratio_y = q_ratio.mean(axis=0)

# plt.plot(q_ratio.columns, q_ratio.mean(axis=0), label='Avg Q Ratio')
# plt.title('Average Q Ratio over time (2023 to 2004)')
# plt.xlabel('Year')
# plt.ylabel('Avg Q')
# plt.grid(True)

# plt.show()

# print (q_ratio_y)

# TODO: Correlation, Relationship, Regression, Cluster Analysis 
#       (cross-compare with methods on cited literatures on types of analysis used)