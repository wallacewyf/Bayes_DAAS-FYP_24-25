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

# # Descriptive Statistics
# # ===========================================================================
# GICS Sector: Financials
descriptive.desc('Financials')

# GICS Sector: Information Technology
descriptive.desc('Information Technology')

# Naive Model
# ===========================================================================
# Return on Assets (ROA)
reg.linear_reg(df = wrangle.all,
               measure = 'roa', 
               esg = 'combined')

# Return on Equity (ROE)
reg.linear_reg(df = wrangle.all,
               measure = 'roe', 
               esg = 'combined')


# Financial Sector
# ===========================================================================
# Return on Assets (ROA)
# Model F1
# ROA / ESG
glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roa',
                 esg = 'combined',
                 log_transform = True)

# Model F2
# ROA / E,S,G
glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roa', 
                 esg = 'individual',
                 log_transform=True)

# Return on Equity (ROE)
# Model F3
# ROE / ESG
reg.linear_reg(df = wrangle.finance, 
               measure = 'roe',
               esg = 'combined',
               log_transform = True)

# Model F4 
# ROE / E,S,G
glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roe',
                 esg = 'individual',
                 log_transform = True, 
                 link = None)               # None defaults as Identity for statsmodels.glm


# Technology Sector
# ===========================================================================
# Return on Assets (ROA)
# Model T1
# ROA / ESG
reg.linear_reg(df = wrangle.tech,
               measure = 'roa',
               esg = 'combined',
               log_transform = True)

# Model T2
# ROA / E,S,G
reg.linear_reg(df = wrangle.tech,
               measure = 'roa',
               esg = 'individual',
               log_transform = True)

# Return on Equity (ROE)
# Model T3
# ROE / ESG
reg.linear_reg(df = wrangle.tech,
               measure = 'roe',
               esg = 'combined',
               log_transform = True)

# Model T4
# ROE / E,S,G
reg.linear_reg(df = wrangle.tech,
               measure = 'roe',
               esg = 'individual',
               log_transform = True)


# # Diagnostic Tests
# # ===========================================================================
data_dfs = [wrangle.finance, wrangle.tech]
measure_types = ['roa', 'roe']
esg_types = ['combined', 'individual']

for data in data_dfs:
    for measure in measure_types:
        for esg in esg_types:
            print ("Basic Linear Regression")
            print(reg.linear_reg(df = data,
                                measure = measure,
                                esg = esg))

            print ("2-Year Lagged Linear Regression")
            print(reg.linear_reg(df = data,
                                measure = measure, 
                                esg = esg, 
                                n_shift = 2))

            print ("Post-2008 Linear Regression")
            print(reg.linear_reg(df = data,
                                measure = measure, 
                                esg = esg, 
                                year_threshold = 2008))

            print ("Gaussian GLM with log-transformation")
            print(glm.gaussian_glm(df = data, 
                                    measure = measure, 
                                    esg = esg, 
                                    log_transform = True))
            
            print ("Gaussian GLM with log-link")
            print(glm.gaussian_glm(df = data, 
                                    measure = measure, 
                                    esg = esg, 
                                    link = 'Log'))

            print ("Tweedie GLM")
            print(glm.tweedie_glm(df = data, 
                                    measure = measure, 
                                    esg = esg))
             
            print ()


stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")