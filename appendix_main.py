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

stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")