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

# # Descriptive Statistics
# # ===========================================================================
# # Available Sectors: 
# #     Financials, Utilities, Health Care, 
# #     Information Technology, Materials, Industrials, 
# #     Energy, Consumer Staples, Consumer Discretionary, 
# #     Real Estate, Communication Services, All

# # GICS Sector: Finance
# descriptive.desc('financials')

# # GICS Sector: Information Technology
# descriptive.desc('Information Technology')

# # Descriptive Statistics for each sector
# industry_arr = ['Financials', 'Utilities', 'Health Care', 
#                 'Information Technology', 'Materials', 'Industrials', 
#                 'Energy', 'Consumer Staples', 'Consumer Discretionary', 
#                 'Real Estate', 'Communication Services']

# for industry in industry_arr:
#     descriptive.desc(industry)

# # Linear Regression
# # ===========================================================================
# reg.linear_reg(df = wrangle.finance, 
#            measure = 'roa', 
#            esg = 'individual')

# reg.linear_reg(df = wrangle.finance, 
#            measure = 'roa', 
#            esg = 'individual',
#            log_transform=True)

# reg.linear_reg(df = wrangle.finance, 
#            measure = 'roa', 
#            esg = 'combined', 
#            year_threshold=2008, 
#            n_shift = None,
#            log_transform=True,
#            cov_type='hc2')


# # Gaussian GLM Regression
# # ===========================================================================
# glm.gaussian_glm(df = wrangle.finance,
#                  measure = 'roa', 
#                  esg = 'individual',
#                  log_transform=True)

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
#                  measure = 'roa', 
#                  esg = 'combined',
#                  year_threshold = None, 
#                  log_transform = True, 
#                  n_shift = None), 
#                  link = 'Log')

# # Gamma GLM Regression
# # ===========================================================================
# Note: the parameters requires further adjustments as extreme outliers in financial returns 
#       may cause imbalanced weights, which then causes infeasible estimations.

# glm.gamma_glm(df = wrangle.finance, 
#               measure = 'roa', 
#               esg = 'individual',
#               year_threshold = None, 
#               log_transform = True, 
#               n_shift = None, 
#               link = 'Log')


# # Tweedie GLM Regression
# # ===========================================================================
# Note: the parameters requires further adjustments as extreme outliers in financial returns 
#       may cause imbalanced weights, which then causes infeasible estimations.

# glm.tweedie_glm(df = wrangle.finance,
#                  measure = 'roa', 
#                  esg = 'combined',
#                  year_threshold = None,
#                  log_transform = True, 
#                  n_shift = None, 
#                  link = None, 
#                  var_power = None)

# # Codespace
# # =======================================================================================
# # Financial Sector
# # Return on Assets (ROA)
# # ROA / ESG
# glm.gamma_glm(df = wrangle.finance, 
#               measure = 'roa',
#               esg = 'combined',
#               log_transform = True, 
#               link = 'Log')

# # # ROA / E,S,G
# glm.gamma_glm(df = wrangle.finance, 
#               measure = 'roa',
#               esg = 'individual',
#               log_transform = True, 
#               link = 'Log')

# # Return on Equity (ROE)
# # # ROE / ESG
# reg.linear_reg(df = wrangle.finance, 
#                measure = 'roe',
#                esg = 'combined',
#                log_transform = True)

# # # ROE / E,S,G
# glm.gaussian_glm(df = wrangle.finance, 
#                  measure = 'roe',
#                  esg = 'individual',
#                  log_transform = True, 
#                  link = None)               # None defaults as Identity for statsmodels.glm


# # Technology Sector
# # ROA / ESG
# reg.linear_reg(df = wrangle.tech, 
#                esg = 'combined', 
#                measure = 'roa',
#                log_transform=True)

# # # # ROA / E,S,G
# reg.linear_reg(df = wrangle.tech, 
#                esg = 'individual', 
#                measure = 'roa',
#                log_transform=True)

# # # # ROE / ESG
# reg.linear_reg(df = wrangle.tech, 
#                esg = 'combined', 
#                measure = 'roe',
#                log_transform=True)

# # # # ROE / E,S,G
# reg.linear_reg(df = wrangle.tech, 
#                esg = 'individual', 
#                measure = 'roe',
#                log_transform=True)



# Tests used to check which is best
# # =======================================================================================
# # GICS Financials
# # # ROA / ESG
# reg.linear_reg(df = wrangle.finance, 
#                esg = 'combined', 
#                measure = 'roa')

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'combined', 
#                measure = 'roa',
#                n_shift = 2)

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'combined', 
#                measure = 'roa', 
#                year_threshold = 2008)

# glm.gaussian_glm(df = wrangle.finance, 
#                  esg = 'combined', 
#                  measure = 'roa', 
#                  link='Log')

# glm.gamma_glm(df = wrangle.finance, 
#                  esg = 'combined', 
#                  measure = 'roa')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'combined', 
#                 measure = 'roa')

# # # ROE / ESG
# reg.linear_reg(df = wrangle.finance, 
#                esg = 'combined', 
#                measure = 'roe')

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'combined', 
#                measure = 'roe',
#                n_shift = 2)

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'combined', 
#                measure = 'roe', 
#                year_threshold = 2008)

# glm.gaussian_glm(df = wrangle.finance, 
#                  esg = 'combined', 
#                  measure = 'roe', 
#                  link='Log')

# glm.gamma_glm(df = wrangle.finance, 
#                  esg = 'combined', 
#                  measure = 'roe')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'combined', 
#                 measure = 'roe')

# # # ROA / E_S_G
# reg.linear_reg(df = wrangle.finance, 
#                esg = 'individual', 
#                measure = 'roa')

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'individual', 
#                measure = 'roa',
#                n_shift = 2)

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'individual', 
#                measure = 'roa', 
#                year_threshold = 2008)

# glm.gaussian_glm(df = wrangle.finance, 
#                  esg = 'individual', 
#                  measure = 'roa', 
#                  link='Log')

# glm.gamma_glm(df = wrangle.finance, 
#                  esg = 'individual', 
#                  measure = 'roa')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'individual', 
#                 measure = 'roa')


# # # ROE / E_S_G
# reg.linear_reg(df = wrangle.finance, 
#                esg = 'individual', 
#                measure = 'roe')

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'individual', 
#                measure = 'roe',
#                n_shift = 2)

# reg.linear_reg(df = wrangle.finance, 
#                esg = 'individual', 
#                measure = 'roe', 
#                year_threshold = 2008)

# glm.gaussian_glm(df = wrangle.finance, 
#                  esg = 'individual', 
#                  measure = 'roe', 
#                  link='Log')

# glm.gamma_glm(df = wrangle.finance, 
#                  esg = 'individual', 
#                  measure = 'roe')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'individual', 
#                 measure = 'roe')


# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'combined', 
#                 measure = 'roa')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'combined', 
#                 measure = 'roe')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'individual', 
#                 measure = 'roa')

# glm.tweedie_glm(df = wrangle.finance, 
#                 esg = 'individual', 
#                 measure = 'roe')



# # GICS: Information Technology
# # =======================================================================================
# glm.gaussian_glm(df = wrangle.tech, 
#                  esg = 'combined', 
#                  measure = 'roa', 
#                  log_transform=True)

# glm.gaussian_glm(df = wrangle.tech, 
#                  esg = 'combined', 
#                  measure = 'roe', 
#                  log_transform=True)


# glm.gaussian_glm(df = wrangle.tech, 
#                  esg = 'individual', 
#                  measure = 'roa', 
#                  log_transform=True)

# glm.gaussian_glm(df = wrangle.tech, 
#                  esg = 'individual', 
#                  measure = 'roe', 
#                  log_transform=True)
