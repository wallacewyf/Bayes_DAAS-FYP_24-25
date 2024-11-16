import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score                        # R-squared
import matplotlib.pyplot as plt

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

# TRY FTSE 350 FIRST
# inner join
esg_scores, q_ratio = ftse_scores.esg_scores.align(ftse_returns.q_ratio, join='inner') 

# inner join
join_df = pd.concat([ftse_scores.e_pillars, ftse_scores.s_pillars, ftse_scores.g_pillars], axis=1, join='inner')

e_pillars = join_df.iloc[:, :20]
s_pillars = join_df.iloc[:, 20:40]
g_pillars = join_df.iloc[:, 40:60]

# correlation analysis
esg_corr = esg_scores.iloc[:, 0].corr(q_ratio.iloc[:, 0])
e_corr = e_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])
s_corr = s_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])
g_corr = g_pillars.iloc[:, 0].corr(q_ratio.iloc[:, 0])

print ('Correlation Analysis')
print ('--------------------------------------------------------------------')
print(f'Correlation between ESG scores and Q ratio: {round(esg_corr, 5)}')
print(f'Correlation between E scores and Q ratio: {round(e_corr, 5)}')
print(f'Correlation between S scores and Q ratio: {round(s_corr, 5)}')
print(f'Correlation between G scores and Q ratio: {round(g_corr, 5)}')


# line graph for relationship between average of Q ratio and ESG scores
esg_avg = esg_scores.mean(axis=0)
q_avg = q_ratio.mean(axis=0)

years = np.arange(2023, 2003, -1)
figure, ax1 = plt.subplots(figsize=(10, 6))

# primary axis
ax1.plot(years, esg_avg, label='Average ESG Scores', color='blue')
ax1.set_xlabel('Year')
ax1.set_ylabel('Average ESG Scores', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(np.arange(2024, 2002, -2))

# secondary axis
ax2 = ax1.twinx()
ax2.plot(years, q_avg, label="Average Tobin's Q", color='green')
ax2.set_ylabel("Average Tobin's Q", color='green')
ax2.tick_params(axis='y', labelcolor='green')

# title, legend, label
plt.title('Yearly Trends of ESG Scores and Tobin\'s Q')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

plt.show()

