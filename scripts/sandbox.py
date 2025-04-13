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

# Technology Sector
# ===========================================================================
# Return on Equity (ROE)
# Model T3
# ROE / ESG
val, a, b = glm.tweedie_glm(df = wrangle.finance,
               measure = 'roe',
               esg = 'combined',
               var_power=1)

print (val, a, b)

val, a, b = glm.gaussian_glm(df = wrangle.finance,
               measure = 'roe',
               esg = 'combined',
               log_transform=True,
               link=None)

print (val, a, b)

val, a, b = reg.linear_reg(df = wrangle.finance,
               measure = 'roe',
               esg = 'combined',
               log_transform=True)

print (val, a, b)


stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")