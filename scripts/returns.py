# import config 

import config

# import libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt


returns_df = pd.read_excel(config.returns)
