import analysis
import timeit
import sys, os
import pandas as pd

import config, wrangle

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

start = timeit.default_timer()

# print (wrangle.scores(config.ftse_esg, 'esg'))

esg_desc_df = analysis.desc(wrangle.scores(config.ftse_esg, 'esg'), 'macro')
stop = timeit.default_timer()

print (esg_desc_df)
print ('\n\n' + 'Time taken: ' + str(stop-start))

# File names (to be moved to config file most likely)
# ESG Descriptive over time

filename = 'ESG Descriptive over time' 
type = 'esg'

with open (config.results_path + filename + '.csv', 'w') as file:
    file.write ('ESG Scores from 2023 to 2004 (excluding where values = 0)')
    file.write ('\n')

esg_desc_df.to_csv(config.results_path + filename + '.csv',
                mode = 'a',
                index=True,
                header=True)
