# packages
import sys
import os
import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score                        # R-squared
from sklearn.preprocessing import PolynomialFeatures

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

# TODO: check if this is the best way to do a join between ESG scores?
# maybe it's better if we just do between ESG, E, S, G instead of ESG and Q

# Descriptive Statistics
# provide descriptive based on cleaned data
# option to stack together for macro or '' for detailed
def desc(df, macro):
    try:
        df = df.mask(df == 0).describe().iloc[1:, :]

        if macro:
            df = df.stack().describe()
            df = df.iloc[1:]

            return df
        
        else: return df
    
    except ValueError as error_msg:
        print ('Check your dataframe and type if it''s valid.')
        print ('Error message:', error_msg)

def output(df, index, type, macro):    
    if type == 'esg': type = 'ESG Scores'
    elif type == 'e': type = 'E Scores'
    elif type == 's': type = 'S Scores'
    elif type == 'g': type = 'G Scores'
    elif type == 'roe': type = 'Return on Equity (ROE)'
    elif type == 'roa': type = 'Return on Assets (ROA)'
    elif type == 'yoy': type = '52 Week Return'
    elif type == 'q': type = 'Q Ratio'

    filename = index.upper() + ' '

    if macro: filename = filename + type + ' Macro Overview'
    else: filename = filename + type + ' Descriptive Overview'

    with open (config.results_path + filename + '.csv', 'w') as file:
        file.write (type + ' Descriptive Statistics from 2023 to 2004 \n')

    df.to_csv(config.results_path + filename + '.csv',
                mode = 'a',
                index=True,
                header=True)

scoring_type = ['esg', 'e', 's', 'g']
finp_type = ['roe', 'roa', 'yoy', 'q']
index_arr = ['msci', 
             'nasdaq', 
            'snp', 
            'stoxx', 
            'ftse']

start = timeit.default_timer()

for each_index in index_arr:
    for each_type in scoring_type:
        output(
            df = desc(
                df = wrangle.scores(
                    each_index, 
                    each_type
                ), 
                macro=False
            ), 
            index = each_index,
            type = each_type,
            macro=False
        )

        output(
            df = desc(
                df = wrangle.scores(
                    each_index, 
                    each_type
                ), 
                macro=True
            ), 
            index = each_index,
            type = each_type,
            macro = True
        )

stop = timeit.default_timer()

print ('Time taken:', str(stop-start), '\n\n')

# split between pre/post COVID pandemic / Paris Agreement
# print (esg_sum.loc[:, [2023, 2022, 2021, 2020, 2019, 2017, 2016, 2015, 2014, 2013, 2008, 2004]])

# --------------------------------------------------------------------
# Regression Analysis 
# notes: 
#   - there probably isn't any linearity
#   - but fix the Q ratio value, then figure out again if we should use r2 or OLS to 
#     to calculate linearity

# r2_values = []
# years = list(range(2004, 2024))

# for year in years:
#     # Extract data for the current year
#     X = q_ratio[year].values.reshape(-1, 1)  # Q Ratio for the year
#     y = esg_scores[year].values             # ESG Scores for the year

#     # Fit a linear regression model
#     model = LinearRegression()
#     model.fit(X, y)
#     y_pred = model.predict(X)

#     # Calculate R² (linearity measure)
#     r2 = r2_score(y, y_pred)
#     r2_values.append(r2)

# print (r2_values)

# # Plot R² values over time
# plt.figure(figsize=(10, 6))
# plt.plot(years, r2_values, marker='o', label='R² (Linearity Measure)')
# plt.title("Linearity (R²) Between ESG Scores and Q Ratio Over Time")
# plt.xlabel("Year")
# plt.ylabel("R² Value")
# plt.grid()
# plt.legend()
# plt.show()

# --------------------------------------------------------------------

# Correlation Analysis -- this can only be done if you know if the relationship is linear/non-linear
# correlation analysis -- this is only for 2023, it has to be time series as esg scores evolves over time
# esg_corr = esg_scores.iloc[:, 0].corr(q_ratio.iloc[:, 0])
# e_corr = e_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])
# s_corr = s_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])
# g_corr = g_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])

# esg_corr = esg_scores.values.flatten()
# q_corr = q_ratio.values.flatten()

# pearson_corr = np.corrcoef(q_corr, esg_corr)[0,1]
# yearly_corr = q_ratio.corrwith(esg_scores, axis=0)

# print (yearly_corr)

# print ('Correlation Analysis')
# print ('--------------------------------------------------------------------')
# print (f'Pearson product-moment correlation coefficient: {pearson_corr}')

# print(f'Correlation between ESG scores and Q ratio: {round(esg_corr, 5)}')
# print(f'Correlation between E scores and Q ratio: {round(e_corr, 5)}')
# print(f'Correlation between S scores and Q ratio: {round(s_corr, 5)}')
# print(f'Correlation between G scores and Q ratio: {round(g_corr, 5)}')

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