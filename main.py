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
llh, aic, bic = reg.linear_reg(df = wrangle.all,
                measure = 'roa', 
                esg = 'combined')

print ("===========================================================================")
print ("Naive Model: ROA")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Return on Equity (ROE)
llh, aic, bic = reg.linear_reg(df = wrangle.all,
               measure = 'roe', 
               esg = 'combined')

print ("===========================================================================")
print ("Naive Model: ROE")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Financial Sector
# ===========================================================================
# Return on Assets (ROA)
# Model F1
# ROA ~ ESG + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roa',
                 esg = 'combined',
                 log_transform = True, 
                 link = None)               # None defaults as Identity for statsmodels.glm

print ("===========================================================================")
print ("Model F1 - Gaussian GLM: ROA ~ ESG + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model F2
# ROA ~ E + S + G + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roa', 
                 esg = 'individual',
                 log_transform=True, 
                 link = None)               # None defaults as Identity for statsmodels.glm

print ("Model F2 - Gaussian GLM: ROA ~ E + S + G + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model F2 - Environmental Pillar
# ROA ~ E + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roa', 
                 esg = 'env_q',
                 log_transform=True, 
                 link = None)               # None defaults as Identity for statsmodels.glm

print ("Model F2 - Gaussian GLM: ROA ~ E + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Return on Equity (ROE)
# Model F3
# ROE ~ ESG + Q_Ratio
llh, aic, bic = reg.linear_reg(df = wrangle.finance, 
               measure = 'roe',
               esg = 'combined',
               log_transform = True)

print ("Model F3 - Linear Regression: ROE ~ ESG + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model F4 
# ROE ~ E + S + G + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roe',
                 esg = 'individual',
                 log_transform = True, 
                 link = None)               # None defaults as Identity for statsmodels.glm

print ("Model F4 - Gaussian GLM: ROE ~ E + S + G + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model F4 - Environmental Pillar
# ROE ~ E + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roe',
                 esg = 'env_q',
                 log_transform = True, 
                 link = None)               # None defaults as Identity for statsmodels.glm

print ("Model F4 - Gaussian GLM: ROE ~ E + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Technology Sector
# ===========================================================================
# Return on Assets (ROA)
# Model T1
# ROA ~ ESG + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roa',
               esg = 'combined',
               var_power=1)

print ("===========================================================================")
print ("Model T1 - Tweedie GLM: ROA ~ ESG + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model T1 (ESG)
# ROA ~ ESG
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roa',
               esg = 'esg',
               var_power=1)


print ("Model T1 - Tweedie GLM: ROA ~ ESG")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model T2
# ROA ~ E + S + G + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roa',
               esg = 'individual',
               var_power=1)

print ("Model T2 - Tweedie GLM: ROA ~ E + S + G + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Best Model T2
# ROA ~ E + S 
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roa',
               esg = 'env_soc',
               var_power=1)

print ("Model T2 - Tweedie GLM: ROA ~ E + S")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Return on Equity (ROE)
# Model T3
# ROE ~ ESG + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roe',
               esg = 'combined',
               var_power=1)

print ("Model T3 - Tweedie GLM: ROE ~ ESG + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Model T4
# ROE ~ E + S + G + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roe',
               esg = 'individual',
               var_power=1)

print ("Model T4 - Tweedie GLM: ROE ~ E + S + G + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

# Best Model T4
# ROA ~ E + S + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech,
               measure = 'roe',
               esg = 'env_soc_q',
               var_power=1)

print ("Model T4 - Tweedie GLM: ROA ~ E + S + Q_Ratio")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")
print ('===========================================================================')

# # # Diagnostic Tests
# # # ===========================================================================
# data_dfs = [wrangle.finance, wrangle.tech]
# measure_types = ['roa', 'roe']
# esg_types = ['combined', 'individual']

# for data in data_dfs:
#     for measure in measure_types:
#         for esg in esg_types:
#             if data.equals(wrangle.finance): print (f'Industry: Finance | Measure: {measure.upper()} | ESG: {esg.title()}')
#             else: print (f'Industry: Technology | Measure: {measure.upper()} | ESG: {esg.title()}')
#             print ()

#             print ("Basic Linear Regression")
#             llh, aic, bic = reg.linear_reg(df = data,
#                                 measure = measure,
#                                 esg = esg)
            
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ("2-Year Lagged Linear Regression")
#             llh, aic, bic = reg.linear_reg(df = data,
#                                 measure = measure, 
#                                 esg = esg, 
#                                 n_shift = 2)
            
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ("Post-2008 Linear Regression")
#             llh, aic, bic = reg.linear_reg(df = data,
#                                 measure = measure, 
#                                 esg = esg, 
#                                 year_threshold = 2008)
                        
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             if data.equals(wrangle.tech):
#                 print ("Gaussian GLM with log-transformation")
#                 llh, aic, bic = glm.gaussian_glm(df = data, 
#                                         measure = measure, 
#                                         esg = esg, 
#                                         log_transform = True)

#                 print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             else:
#                 print ("Gaussian GLM with log-link")
#                 llh, aic, bic = glm.gaussian_glm(df = data, 
#                                         measure = measure, 
#                                         esg = esg, 
#                                         link = 'Log')

#                 print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ("Tweedie GLM")
#             llh, aic, bic = glm.tweedie_glm(df = data, 
#                                     measure = measure, 
#                                     esg = esg)
            
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ()

#             print ('===========================================================================')

# Models in Chapter 8: Appendix
# ===================================================================================================================
# # Financials 
# ROA ~ E + S + G
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                                 esg = 'env_soc_gov', 
                                 measure = 'roa', 
                                 log_transform=True,
                                 link=None)

print ("Financials - Gaussian GLM: ROA ~ E + S + G")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROA ~ E + S + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                                 esg = 'env_soc_q', 
                                 measure = 'roa', 
                                 log_transform=True,
                                 link=None)

print ("Financials - Gaussian GLM: ROA ~ E + S + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROA ~ E + G + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                                 esg = 'env_gov_q', 
                                 measure = 'roa', 
                                 log_transform=True,
                                 link=None)

print ("Financials - Gaussian GLM: ROA ~ E + G + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROE ~ E + S + G
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                                 esg = 'env_soc_gov', 
                                 measure = 'roe', 
                                 log_transform=True,
                                 link=None)

print ("Financials - Gaussian GLM: ROE ~ E + S + G")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROE ~ E + S + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                                 esg = 'env_soc_q', 
                                 measure = 'roe', 
                                 log_transform=True,
                                 link=None)

print ("Financials - Gaussian GLM: ROE ~ E + S + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROE ~ E + G + Q_Ratio
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                                 esg = 'env_gov_q', 
                                 measure = 'roe', 
                                 log_transform=True,
                                 link=None)

print ("Financials - Gaussian GLM: ROE ~ E + G + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ===================================================================================================================
# # Technology
# ROA ~ E + S + G
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech, 
                                 esg = 'env_soc_gov', 
                                 measure = 'roa')

print ("Technology - Tweedie GLM: ROA ~ E + S + G")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROA ~ E + S + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech, 
                                 esg = 'env_soc_q', 
                                 measure = 'roa')

print ("Technology - Tweedie GLM: ROA ~ E + S + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROA ~ E + G + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech, 
                                 esg = 'env_gov_q', 
                                 measure = 'roa')

print ("Technology - Tweedie GLM: ROA ~ E + G + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROE ~ E + S + G
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech, 
                                 esg = 'env_soc_gov', 
                                 measure = 'roe')

print ("Technology - Tweedie GLM: ROE ~ E + S + G")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')

# ROE ~ E + G + Q_Ratio
llh, aic, bic = glm.tweedie_glm(df = wrangle.tech, 
                                 esg = 'env_gov_q', 
                                 measure = 'roe')

print ("Technology - Tweedie GLM: ROE ~ E + G + Q_Ratio")
print (f'Log-likelihood: {round(llh, 2)} | AIC: {round(aic, 2)} | BIC: {round(bic, 2)}')


# # # Diagnostic Tests
# # # ===========================================================================
# data_dfs = [wrangle.finance, wrangle.tech]
# measure_types = ['roa', 'roe']
# esg_types = ['combined', 'individual']

# for data in data_dfs:
#     for measure in measure_types:
#         for esg in esg_types:
#             if data.equals(wrangle.finance): print (f'Industry: Finance | Measure: {measure.upper()} | ESG: {esg.title()}')
#             else: print (f'Industry: Technology | Measure: {measure.upper()} | ESG: {esg.title()}')
#             print ()

#             print ("Basic Linear Regression")
#             llh, aic, bic = reg.linear_reg(df = data,
#                                 measure = measure,
#                                 esg = esg)
            
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ("2-Year Lagged Linear Regression")
#             llh, aic, bic = reg.linear_reg(df = data,
#                                 measure = measure, 
#                                 esg = esg, 
#                                 n_shift = 2)
            
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ("Post-2008 Linear Regression")
#             llh, aic, bic = reg.linear_reg(df = data,
#                                 measure = measure, 
#                                 esg = esg, 
#                                 year_threshold = 2008)
                        
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             if data.equals(wrangle.tech):
#                 print ("Gaussian GLM with log-transformation")
#                 llh, aic, bic = glm.gaussian_glm(df = data, 
#                                         measure = measure, 
#                                         esg = esg, 
#                                         log_transform = True)

#                 print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             else:
#                 print ("Gaussian GLM with log-link")
#                 llh, aic, bic = glm.gaussian_glm(df = data, 
#                                         measure = measure, 
#                                         esg = esg, 
#                                         link = 'Log')

#                 print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ("Tweedie GLM")
#             llh, aic, bic = glm.tweedie_glm(df = data, 
#                                     measure = measure, 
#                                     esg = esg)
            
#             print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")

#             print ()

#             print ('===========================================================================')

stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")