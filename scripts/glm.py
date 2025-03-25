# packages
import sys, os, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# statistical packages
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels import api as sm 
from statsmodels.formula import api as smf
from scipy import stats

import linear_reg as reg 

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# other path
import config, wrangle

# initialize logger module 
log = config.logging
 
# VIF Calculation available as reg.vif(X)

def basic_gaussian(df, measure, type):
    data = df
    industry = data.index.get_level_values(1).unique()[0]
    measure = measure.upper()

    # Set independent variables to be ESG or E,S,G according to type
    if type == 1:
        X = data[['ESG', 'Q_Ratio']]
        eqn = f"{measure} ~ ESG + Q_Ratio"

        if len(data.index.get_level_values(2).unique()):
            output_path = config.threshold_basic_glm + f'{industry}/{measure}/ESG/'
        
        else: output_path = config.basic_glm + f'{industry}/{measure}/ESG/'

    elif type == 2:
        X = data[['E', 'S', 'G', "Q_Ratio"]]
        eqn = f"{measure} ~ E + S + G + Q_Ratio"
        output_path = config.basic_glm + f'{industry}/{measure}/E_S_G/'

    glm = smf.glm(eqn, 
                  data=data,
                  family=sm.families.Gaussian()).fit()
    
    print (glm.summary())
    

def init_data(df, measure, type):
    data = df
    industry = data.index.get_level_values(1).unique()[0]
    measure = measure.upper()

    if type == 1:
        eqn = f"{measure} ~ ESG + Q_Ratio"

        if len(data.index.get_level_values(2).unique()):
            output_path = config.threshold_basic_glm + f'{industry}/{measure}/ESG/'
        
        else: output_path = config.basic_glm + f'{industry}/{measure}/ESG/'

    elif type == 2:
        eqn = f"{measure} ~ E + S + G + Q_Ratio"
        output_path = config.basic_glm + f'{industry}/{measure}/E_S_G/'
        
    


# Codespace 
# =======================================================

# basic_gaussian(df = wrangle.finance, 
#          measure = 'roe', 
#          type = 1)