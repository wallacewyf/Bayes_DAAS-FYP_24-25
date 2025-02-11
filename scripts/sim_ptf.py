# packages
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# import packages
import config, wrangle, descriptive

# TODO:
# 1. Split each category into quartiles in each index.
# 2. Run simulation on top 25%, bottom 25%, and index-linked fund performance as 3 separate portfolios.
# 3. Repeat for each index.
# 4. Create data visualization to compare performance.

# Naming convention:
# Q1 - Lowest 25%
# Q4 - Highest 25%

# avg_company_scores = descriptive.desc(wrangle.scores('nasdaq', 'esg').T, macro=False).iloc[1,:]
# avg_company_scores.sort_values(ascending=False, inplace=True)
# avg_ident = list(avg_company_scores.index.get_level_values(0))
# avg = list(avg_company_scores.index.get_level_values(1))

# q1_ident = avg_ident[0:int(round((len(avg)/4),0))]
# q4_ident = avg_ident[int(round((len(avg)/4*3),0)):len(avg)][::-1]

# q1_companies = avg[0:int(round((len(avg)/4),0))]
# q4_companies = avg[int(round((len(avg)/4*3),0)):len(avg)][::-1]

# TODO: data validation for LEN(Q1_COMPANIES) == LEN(Q4_COMPANIES)

# -------------------------------------------------------------------------------------------------------------------------------------

roe_df = wrangle.returns('nasdaq', 'roe')
roe_5y = roe_df.iloc[:, :5]

nasdaq_esg = wrangle.scores('nasdaq', 'esg')
esg_5y = nasdaq_esg.iloc[:, :5]

esg = roe_df.index.intersection(esg_5y.index)
esg_5y = esg_5y.reindex(esg)
roe_5y = roe_df.iloc[:, :5].reindex(esg)

# esg_index = set(esg_5y.index.get_level_values(1))
# roe_index = set(roe_5y.index.get_level_values(1))

# key_1 = roe_index.intersection(esg_index)
# key_2 = esg_index.intersection(roe_index)

# scope = sorted(list(set(key_1.intersection(key_2))))
# cond = roe_5y.index.get_level_values('Company Name').isin(scope)

# roe_5y = roe_5y[cond]
# esg_5y = esg_5y[cond]

# # -------------------------------------------------------------------------------------------------------------------------------------

avg_company_scores = esg_5y
# avg_company_scores.dropna(inplace=True)

avg_company_scores = descriptive.desc(avg_company_scores.T, macro=False).iloc[1,:]
avg_company_scores.sort_values(ascending=False, inplace=True)
avg_ident = list(avg_company_scores.index.get_level_values(0))
avg = list(avg_company_scores.index.get_level_values(1))

q1_ident = avg_ident[0:int(round((len(avg)/4),0))]
q4_ident = avg_ident[int(round((len(avg)/4*3),0)):len(avg)][::-1]

q1_companies = avg[0:int(round((len(avg)/4),0))]
q4_companies = avg[int(round((len(avg)/4*3),0)):len(avg)][::-1]

q1_cond = roe_5y.index.get_level_values('Company Name').isin(q1_companies)
q1_df = roe_5y[q1_cond]
q1_df.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
q1_df = q1_df.T
q1_df *= 100

q4_cond = roe_5y.index.get_level_values('Company Name').isin(q4_companies)
q4_df = roe_5y[q4_cond]
q4_df.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)


# TODO: once I get the intersection, there's still an existing problem where NaN will still be picked up 
# I'm guessing its because of the .isin scope that's causing this issue
# If we use the keys from the set intersection earlier above - it should only pick up companies without NaNs.

# update: resolved using reindex (but to confirm if dropna is really needed in wrangle.py)

print (q1_df)
print (q4_df)
print (roe_5y)

# TODO: plot roe returns on graph

for each_company in q1_df.columns:
    plt.plot(q1_df.index, q1_df[each_company], marker="x", label=each_company)

plt.xlabel ('Year')
plt.ylabel ('Return on Equity (%)')
plt.title ('ROE % Trends (2019 - 2023)')

plt.xticks(q1_df.index)  # Ensure correct year spacing

plt.legend(title="Company")
# plt.gca().invert_xaxis()  # Reverse X-axis to show most recent year first
plt.grid(True)
plt.show()
