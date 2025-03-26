# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.path.join(os.path.dirname(__file__), "scripts")
sys.path.append(data_path)

import descriptive
import linear_reg as reg
import glm

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

start = datetime.datetime.now()

# Data Wrangling
# ===========================================================================
# Available dataframes:
# Financials = wrangle.finance 
# Utilities = wrangle.utils 
# Health Care = wrangle.health
# Information Technology = wrangle.tech
# Materials = wrangle.materials 
# Industrials = wrangle.industrials 
# Energy = wrangle.energy 
# Consumer Staples = wrangle.consumer_staple
# Consumer Discretionary = wrangle.consumer_disc
# Real Estate = wrangle.reits
# Communication Services = wrangle.comm

# Finance Sector
# print (wrangle.finance)

# Technology Sector
# print (wrangle.tech)

# Descriptive Statistics
# ===========================================================================
# Available Sectors: 
    # Financials, Utilities, Health Care, 
    # Information Technology, Materials, Industrials, 
    # Energy, Consumer Staples, Consumer Discretionary, 
    # Real Estate, Communication Services, All

# GICS Sector: Finance
# descriptive.desc('financials')

# # Descriptive Statistics for each sector
# industry_arr = ['Financials', 'Utilities', 'Health Care', 
#                 'Information Technology', 'Materials', 'Industrials', 
#                 'Energy', 'Consumer Staples', 'Consumer Discretionary', 
#                 'Real Estate', 'Communication Services']

# for industry in industry_arr:
#     descriptive.desc(industry)

# # Linear Regression
# # ===========================================================================
# # Finance
# reg.linear_reg(df = wrangle.finance,
#            measure = 'roe', 
#            type = 1)


# reg.linear_reg(df = wrangle.finance,
#            measure = 'ROA', 
#            type = 2)

# # Technology
# reg.linear_reg(df = wrangle.tech,
#            measure = 'Roe', 
#            type = 1)


# reg.linear_reg(df = wrangle.tech,
#            measure = 'roA', 
#            type = 2)

# # Lagged Linear Regression
# # ===========================================================================
# # Finance
# reg.lagged_reg(df = wrangle.finance, 
#            measure = 'roe', 
#            type = 1,
#            n = 1)

# reg.lagged_reg(df = wrangle.tech, 
#            measure = 'roe', 
#            type = 2,
#            n = 1)

# # Technology
# reg.lagged_reg(df = wrangle.finance, 
#            measure = 'roa', 
#            type = 1,
#            n = 1)

# reg.lagged_reg(df = wrangle.tech, 
#            measure = 'roA', 
#            type = 2,
#            n = 1)

# # Log-Transformed Lagged Linear Regression
# # ===========================================================================
# # Finance
# reg.log_lag(df = wrangle.finance, 
#            measure = 'roe', 
#            type = 1,
#            n = 1)

# reg.log_lag(df = wrangle.tech, 
#            measure = 'roe', 
#            type = 2,
#            n = 1)

# # Technology
# reg.log_lag(df = wrangle.finance, 
#            measure = 'roa', 
#            type = 1,
#            n = 1)

# reg.log_lag(df = wrangle.tech, 
#            measure = 'roA', 
#            type = 2,
#            n = 1)

# # Log-Transformed Linear Regression
# # ===========================================================================
# # Finance
# reg.log_linear(df = wrangle.finance, 
#            measure = 'roe', 
#            type = 1,)

# reg.log_linear(df = wrangle.tech, 
#            measure = 'roe', 
#            type = 2)

# # Technology
# reg.log_linear(df = wrangle.finance, 
#            measure = 'roa', 
#            type = 1)

# reg.log_linear(df = wrangle.tech, 
#            measure = 'roA', 
#            type = 2)

# # Linear Regression after threshold of 2008
# # i.e. if threshold = 2008, data only takes data post-2008 (2009, 2010, ...)
# # ===========================================================================
# # Finance
# reg.threshold_linear(
#     df = wrangle.finance, 
#     measure = 'roe',
#     type = 1, 
#     year = 2008
# )


# reg.threshold_log_linear(
#     df = wrangle.finance, 
#     measure = 'roe',
#     type = 1, 
#     year = 2008
# )


# reg.threshold_lag(
#     df = wrangle.finance, 
#     measure = 'roe',
#     type = 1, 
#     year = 2008,
#     n = 2
# )


# reg.threshold_log_lag(
#     df = wrangle.finance, 
#     measure = 'roe',
#     type = 1, 
#     year = 2008,
#     n = 2
# )

# # Technology
# reg.threshold_linear(
#     df = wrangle.tech, 
#     measure = 'roa',
#     type = 1, 
#     year = 2008
# )


# reg.threshold_log_linear(
#     df = wrangle.tech, 
#     measure = 'roa',
#     type = 1, 
#     year = 2008
# )


# reg.threshold_lag(
#     df = wrangle.tech, 
#     measure = 'roa',
#     type = 1, 
#     year = 2008,
#     n = 2
# )


# reg.threshold_log_lag(
#     df = wrangle.tech, 
#     measure = 'roa',
#     type = 1, 
#     year = 2008,
#     n = 2
# )


# # Gaussian GLM Regression
# # ===========================================================================
# glm.gaussian_glm(df = wrangle.finance,
#                  measure = 'roe', 
#                  esg = 'combined')

# glm.gaussian_glm(df = wrangle.finance, 
#                  measure = 'roa', 
#                  esg = 'combined', 
#                  year_threshold = 2008, 
#                  n_shift=2, 
#                  log_transform=True)


stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")