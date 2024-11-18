# packages
import sys
import os
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
import config

# import returns 
import ftse_returns
# import msci_returns
# import nasdaq_returns
# import snp500_returns
# import stoxx_returns

# import scores
import ftse_scores
# import msci_scores
# import nasdaq_scores
# import snp500_scores
# import stoxx_scores

# FTSE 350
# inner join
merged_df = pd.concat([ftse_scores.esg_scores,
                        ftse_scores.e_pillars, 
                        ftse_scores.s_pillars, 
                        ftse_scores.g_pillars, 
                        ftse_returns.q_ratio], 
                        axis=1,
                        join='inner')

# separate scores and Q ratio into separate dataframes
esg_scores = merged_df.iloc[:, :20]
e_pillars = merged_df.iloc[:, 20:40]
s_pillars = merged_df.iloc[:, 40:60]
g_pillars = merged_df.iloc[:, 60:80]
q_ratio = merged_df.iloc[:, 80:]

# omit values where 0 for a more macro perspective, excludes count
esg_sum = esg_scores.mask(esg_scores == 0).describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
e_sum = e_pillars.mask(e_pillars == 0).describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
s_sum = s_pillars.mask(s_pillars == 0).describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
g_sum = g_pillars.mask(g_pillars == 0).describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

# macro description, excludes count
esg_stack_sum = esg_sum.stack().describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
e_stack_sum = e_sum.stack().describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
s_stack_sum = s_sum.stack().describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
g_stack_sum = g_sum.stack().describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

# # Descriptive Analysis for ESG Scores
# # Time-Series Summary
# print ('Time Series Descriptive Statistic')
# print ('ESG Scores from 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (esg_sum, '\n')

# print ('E Scores from 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (e_sum, '\n')

# print ('S Scores from 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (s_sum, '\n')

# print ('G Scores from 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (g_sum, '\n\n')

# print ('Summary Statistics')
# print ('ESG Scores over 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (esg_stack_sum, '\n')

# print ('E Scores over 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (e_stack_sum, '\n')

# print ('S Scores over 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (s_stack_sum, '\n')

# print ('G Scores over 2023 to 2004 (excluding where values = 0)')
# print ('--------------------------------------------------------------------')
# print (g_stack_sum, '\n')

# --------------------------------------------------------------------
# Regression Analysis 
# notes: 
#   - there probably isn't any linearity
#   - but fix the Q ratio value, then figure out again if we should use r2 or OLS to 
#     to calculate linearity


r2_values = []
years = list(range(2004, 2024))

for year in years:
    # Extract data for the current year
    X = q_ratio[year].values.reshape(-1, 1)  # Q Ratio for the year
    y = esg_scores[year].values             # ESG Scores for the year

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate R² (linearity measure)
    r2 = r2_score(y, y_pred)
    r2_values.append(r2)

print (r2_values)

# Plot R² values over time
plt.figure(figsize=(10, 6))
plt.plot(years, r2_values, marker='o', label='R² (Linearity Measure)')
plt.title("Linearity (R²) Between ESG Scores and Q Ratio Over Time")
plt.xlabel("Year")
plt.ylabel("R² Value")
plt.grid()
plt.legend()
plt.show()

# --------------------------------------------------------------------

# Correlation Analysis -- this can only be done if you know if the relationship is linear/non-linear
# correlation analysis -- this is only for 2023, it has to be time series as esg scores evolves over time
esg_corr = esg_scores.iloc[:, 0].corr(q_ratio.iloc[:, 0])
e_corr = e_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])
s_corr = s_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])
g_corr = g_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])

esg_corr = esg_scores.values.flatten()
q_corr = q_ratio.values.flatten()

pearson_corr = np.corrcoef(q_corr, esg_corr)[0,1]
yearly_corr = q_ratio.corrwith(esg_scores, axis=0)

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

# TODO: Correlation, Relationship, Regression, Cluster Analysis 
#       (cross-compare with methods on cited literatures on types of analysis used)