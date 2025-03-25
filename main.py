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
descriptive.desc('financials')

# # Descriptive Statistics for each sector
# industry_arr = ['Financials', 'Utilities', 'Health Care', 
#                 'Information Technology', 'Materials', 'Industrials', 
#                 'Energy', 'Consumer Staples', 'Consumer Discretionary', 
#                 'Real Estate', 'Communication Services']

# for industry in industry_arr:
#     descriptive.desc(industry)

# Linear Regression
# ===========================================================================

reg.linear_reg(df = wrangle.finance,
           measure = 'roe', 
           type = 1)


reg.linear_reg(df = wrangle.finance,
           measure = 'ROA', 
           type = 2)

# Technology
reg.linear_reg(df = wrangle.tech,
           measure = 'Roe', 
           type = 1)


reg.linear_reg(df = wrangle.tech,
           measure = 'roA', 
           type = 2)


# Gaussian GLM Regression
# ===========================================================================
# regression.gaussian_glm(
#     index='msci', 
#     measure='roe', 
#     scores=True
# )


# regression.gaussian_glm(
#     index='msci', 
#     measure='roe', 
#     scores=False
# )

stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")