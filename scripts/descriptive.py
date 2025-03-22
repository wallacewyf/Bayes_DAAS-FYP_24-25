# packages
import sys, os
from datetime import datetime

# set path to retrieve returns/scores files
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# import packages
import config, wrangle

# initialize logger module 
log = config.logging

# ===========================================================================
# Descriptive Statistics

def desc(industry):
    if industry.lower().startswith("fin"):
        '''
        Descriptive Statistics for Financial Companies 
        '''
        data = wrangle.finance
        output = os.path.join(config.desc_path, 'Descriptive - Financials.csv')
        desc = data.describe()

        log.info (f"Writing Descriptive Summary for Financial Companies")
        with open(output, "w") as finance:
            finance.write(f"Descriptive Statistics for Financial Companies\n")
            finance.write(f"Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n")
            finance.write(str(desc))

    elif industry.lower().startswith("tech"):
        '''
        Descriptive Statistics for Technology Companies
        '''
        data = wrangle.tech
        output = os.path.join(config.desc_path, 'Descriptive - Technology.csv')
        desc = data.describe()

        log.info (f"Writing Descriptive Summary for Technology Companies")
        with open(output, "w") as tech:
            tech.write(f"Descriptive Statistics for Technology Companies\n")
            tech.write(f"Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n")
            tech.write(str(desc))

    elif industry.lower() == 'all':
        '''
        Descriptive Statistics for all industries
        '''
        data = wrangle.df

        output = os.path.join(config.desc_path, 'Descriptive - All Industries.csv')
        desc = data.describe()

        log.info (f"Writing Descriptive Summary for all industries")
        with open(output, 'w') as all: 
            all.write(f"Descriptive Statistics for all industries\n")
            all.write(f"Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n")
            all.write(str(desc))


    return desc

