# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import descriptive, regression

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# config path
import config, wrangle


start = datetime.datetime.now()

# Descriptive Statistics
descriptive.agg_desc(
    type='scores'
)

descriptive.desc(
    index='msci', 
    measure='roe', 
    macro=False
)

# Gaussian GLM Regression
regression.gaussian_glm(
    index='msci', 
    measure='roe', 
    scores=True
)


stop = datetime.datetime.now()

print (f"Runtime: {stop-start}")