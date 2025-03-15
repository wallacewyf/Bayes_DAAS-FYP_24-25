# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = os.path.join(os.path.dirname(__file__), "scripts")
sys.path.append(data_path)

import descriptive, regression

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle

start = datetime.datetime.now()

# Data Wrangling
# ---------------------------------------------------
# wrangle.returns(index = 'msci', 
#                 measure = 'roe')

# wrangle.scores(index = 'msci', 
#                measure = 'esg')

# Descriptive Statistics
# ---------------------------------------------------
# descriptive.agg_desc(type='scores')
# descriptive.agg_desc(type='finp')

# descriptive.desc(
#     index='msci', 
#     measure='roe', 
#     macro=False
# )

# # Gaussian GLM Regression
regression.gaussian_glm(
    index='msci', 
    measure='roe', 
    scores=True
)


# regression.gaussian_glm(
#     index='msci', 
#     measure='roe', 
#     scores=False
# )

df = wrangle.scores(index='msci', measure='esg')
df.to_csv(config.results_path + 'esg.csv')


stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")