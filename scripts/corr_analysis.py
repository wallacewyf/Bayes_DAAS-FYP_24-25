import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score        # R-squared
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

# regression analysis


print ('Correlation Analysis')
print ('--------------------------------------------------------------------')
print(f'Correlation between ESG scores and Q ratio: {round(esg_corr, 5)}')
print(f'Correlation between E scores and Q ratio: {round(e_corr, 5)}')
print(f'Correlation between S scores and Q ratio: {round(s_corr, 5)}')
print(f'Correlation between G scores and Q ratio: {round(g_corr, 5)}')

