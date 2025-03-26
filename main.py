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
descriptive.desc('financials')

# # Descriptive Statistics for each sector
# industry_arr = ['Financials', 'Utilities', 'Health Care', 
#                 'Information Technology', 'Materials', 'Industrials', 
#                 'Energy', 'Consumer Staples', 'Consumer Discretionary', 
#                 'Real Estate', 'Communication Services']

# for industry in industry_arr:
#     descriptive.desc(industry)

# # Linear Regression
# # ===========================================================================
reg.linear_reg(df = wrangle.finance, 
           measure = 'roa', 
           esg = 'combined')


# reg.linear_reg(df = wrangle.finance, 
#            measure = 'roa', 
#            esg = 'combined', 
#            year_threshold=2008, 
#            log_transform=True,
#            cov_type='hc2')


# Gaussian GLM Regression
# ===========================================================================
glm.gaussian_glm(df = wrangle.finance,
                 measure = 'roe', 
                 esg = 'combined')

# glm.gaussian_glm(df = wrangle.finance, 
#                  measure = 'roa', 
#                  esg = 'combined', 
#                  year_threshold = 2008, 
#                  n_shift=2, 
#                  log_transform=True,
#                  link='log')

# glm.gaussian_glm(df = wrangle.finance, 
#                  measure = 'roe', 
#                  esg = 'individual', 
#                  year_threshold = 2008, 
#                  log_transform=True,
#                  link='inverse')

# # Inverse Gaussian GLM Regression
# # ===========================================================================
# Note: the parameters requires further adjustments as extreme outliers in financial returns 
#       may cause imbalanced weights, which then causes infeasible estimations.

# glm.inv_gaussian(df = wrangle.finance,
#                  measure = 'roe', 
#                  esg = 'combined',
#                  year_threshold = None, 
#                  log_transform = None, 
#                  n_shift = None, 
#                  link = 'Log')

# # Gamma GLM Regression
# # ===========================================================================
# Note: the parameters requires further adjustments as extreme outliers in financial returns 
#       may cause imbalanced weights, which then causes infeasible estimations.

# glm.gamma_glm(df = wrangle.finance, 
#               measure = 'roe', 
#               esg = 'combined',
#               year_threshold = None, 
#               log_transform = None, 
#               n_shift = None, 
#               link = 'Log')

# # Tweedie GLM Regression
# # ===========================================================================
# Note: the parameters requires further adjustments as extreme outliers in financial returns 
#       may cause imbalanced weights, which then causes infeasible estimations.

# glm.tweedie_glm(df = wrangle.finance,
#                  measure = 'roe', 
#                  esg = 'combined',
#                  year_threshold = None,
#                  log_transform = None, 
#                  n_shift = None, 
#                  link = None, 
#                  var_power = None)

stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")