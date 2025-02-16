# packages
import sys, os
import pandas as pd
import matplotlib.pyplot as plt

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# import packages
import config, wrangle, descriptive

# -------------------------------------------------------------------------------------------------------------------------------------

# Notes (14/2/2025):
# MSCI - International Index
# the remaining to be geo-specific
# ignore returns and extract average share prices from Yahoo Finance's API 
# extract scope from existing ESG scores retrieved from Eikon 

# simulate portfolio performance based on ESG scores and share prices from Yahoo's API
# with the weighting from the ESG scores
# compare performance against Eikon's ROE/ROA/Q Ratio to compare returns' trendline with measures' trendline

# H0: ESG scores has an impact on the financial performance of companies
# H1: ESG scores has no impact on the financial performance of companies

# Check for corr between ESG scores and ROE ratio for each index and each industry

# TODO:
# 1. Split each category into quartiles in each index.
# 2. Run simulation on top 25%, bottom 25%, and index-linked fund performance as 3 separate portfolios.
# 3. Repeat for each index.
# 4. Create data visualization to compare performance.

# Naming convention:
# Q1 - Lowest 25%
# Q4 - Highest 25%

# -------------------------------------------------------------------------------------------------------------------------------------

def extract_scope(index, years):

    roe_df = wrangle.returns(index, 'roe')
    roe = roe_df.iloc[:, :years]
    # roe_10y = roe_df.iloc[:, :10]
    # roe_20y = roe_df.iloc[:, :20]

    esg_df = wrangle.scores(index, 'esg')
    esg = esg_df.iloc[:, :years]
    # esg_10y = esg_df.iloc[:, :10]
    # esg_20y = esg_df.iloc[:, :20]

    scope = roe_5y.index.intersection(esg_5y.index)
    # scope_10y = roe_10y.index.intersection(esg_10y.index)
    # scope_20y = roe_20y.index.intersection(esg_20y.index)
    
    esg = esg.reindex(scope)
    roe = roe.reindex(scope)

    # esg_10y = esg_10y.reindex(scope_10y)
    # roe_10y = roe_10y.reindex(scope_10y)

    # esg_20y = esg_20y.reindex(scope_20y)
    # roe_20y = roe_20y.reindex(scope_20y)

    return [roe, esg]


def sim_roe(index, years):

    roe_df = wrangle.returns(index, 'roe')
    roe = roe_df.iloc[:, :years]
    # roe_10y = roe_df.iloc[:, :10]
    # roe_20y = roe_df.iloc[:, :20]

    esg_df = wrangle.scores(index, 'esg')
    esg = esg_df.iloc[:, :years]
    # esg_10y = esg_df.iloc[:, :10]
    # esg_20y = esg_df.iloc[:, :20]

    scope = roe.index.intersection(esg.index)
    # scope_10y = roe_10y.index.intersection(esg_10y.index)
    # scope_20y = roe_20y.index.intersection(esg_20y.index)
    
    esg = esg.reindex(scope)
    roe = roe.reindex(scope)

    # roe_10y = dfs[2]
    # esg_10y = dfs[3]

    # roe_20y = dfs[4]
    # esg_20y = dfs[5]

    company_scores_df = esg
    company_scores_df = descriptive.desc(company_scores_df.T, macro=False).iloc[1,:]
    company_scores_df.sort_values(ascending=False, inplace=True)

    avg = list(company_scores_df.index.get_level_values(1))

    q1_companies = avg[0:int(round((len(avg)/4),0))]
    q4_companies = avg[int(round((len(avg)/4*3),0)):len(avg)][::-1]
    
    # Q1 - top 25% quartile
    q1_cond = roe.index.get_level_values('Company Name').isin(q1_companies)
    q1_df = roe[q1_cond]
    q1_df.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
    q1_df *= 100

    q1_roe_yoy = pd.DataFrame(q1_df.describe().iloc[1,])
    q1_roe_yoy.sort_index(ascending=True, inplace=True)
    # print (q1_roe_yoy)

    # Q4 - bottom 25% quartile
    q4_cond = roe.index.get_level_values('Company Name').isin(q4_companies)
    q4_df = roe[q4_cond]
    q4_df.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
    q4_df *= 100

    q4_roe_yoy = pd.DataFrame(q4_df.describe().iloc[1,])
    q4_roe_yoy.sort_index(ascending=True, inplace=True)
    # print (q4_roe_yoy)

    # all companies
    roe.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
    roe *= 100
    roe = roe.describe().iloc[1,:]
    roe.sort_index(ascending=True, inplace=True)

    return [[q1_roe_yoy, q4_roe_yoy, roe], years]

# -------------------------------------------------------------------------------------------------------------------------------------

roe_df = wrangle.returns('nasdaq', 'roe')
roe_5y = roe_df.iloc[:, :5]

nasdaq_esg = wrangle.scores('nasdaq', 'esg')
esg_5y = nasdaq_esg.iloc[:, :5]

esg = roe_df.index.intersection(esg_5y.index)
esg_5y = esg_5y.reindex(esg)
roe_5y = roe_df.reindex(esg)

# esg_index = set(esg_5y.index.get_level_values(1))
# roe_index = set(roe_5y.index.get_level_values(1))

# key_1 = roe_index.intersection(esg_index)
# key_2 = esg_index.intersection(roe_index)

# scope = sorted(list(set(key_1.intersection(key_2))))
# cond = roe_5y.index.get_level_values('Company Name').isin(scope)

# roe_5y = roe_5y[cond]
# esg_5y = esg_5y[cond]

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

# Q1 - top 25% quartile
q1_cond = roe_5y.index.get_level_values('Company Name').isin(q1_companies)
q1_df = roe_5y[q1_cond]
q1_df.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
q1_df *= 100

q1_roe_yoy = pd.DataFrame(q1_df.describe().iloc[1,])
q1_roe_yoy.sort_index(ascending=True, inplace=True)
# print (q1_roe_yoy)

# Q4 - bottom 25% quartile
q4_cond = roe_5y.index.get_level_values('Company Name').isin(q4_companies)
q4_df = roe_5y[q4_cond]
q4_df.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
q4_df *= 100

q4_roe_yoy = pd.DataFrame(q4_df.describe().iloc[1,])
q4_roe_yoy.sort_index(ascending=True, inplace=True)
# print (q4_roe_yoy)

# all companies
roe_5y.reset_index(['Identifier', 'GICS Industry Name', 'Exchange Name'], drop=True, inplace=True)
roe_5y *= 100
roe_5y = roe_5y.describe().iloc[1,:]
roe_5y.sort_index(ascending=True, inplace=True)

# print (roe_5y)

# # TODO: plot roe returns on graph

def plot_roe(df_year):
    q1_roe_yoy = df_year[0][0]
    q4_roe_yoy = df_year[0][1]
    roe_5y = df_year[0][2]

    range = df_year[1]

    plt.plot(q1_roe_yoy, marker="x", label="Top 25% ESG Scores Companies")
    plt.plot(q4_roe_yoy, marker="o", label="Bottom 25% ESG Scores Companies")
    plt.plot(roe_5y, marker="^", label="All Companies")

    plt.xlabel("Year")
    plt.ylabel("Average ROE %")
    plt.title(f"Average ROE % Trends (2019 - 2023)")
    plt.legend(title="Company Group")

    plt.grid (True)
    plt.show()

plot_roe(sim_roe('nasdaq', 5))

# print (sim_roe('nasdaq', 5)[1])

# -------------------------------------------------------------------------------------------------------------------------------------

# print (wrangle.returns('nasdaq', 'mktcap