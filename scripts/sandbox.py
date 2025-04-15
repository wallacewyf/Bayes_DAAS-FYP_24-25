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

# Codespace
# ==========================================

# Model F2 - Environmental and Social Pillar
# ROA / E 
llh, aic, bic = glm.gaussian_glm(df = wrangle.finance, 
                 measure = 'roa', 
                 esg = 'env_soc',
                 log_transform=True, 
                 link = None)               # None defaults as Identity for statsmodels.glm

print ("Model F2 (Environmental and Social):")
print (f"Log-likelihood: {llh} | AIC: {aic} | BIC: {bic}")


stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")