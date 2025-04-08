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
    '''
    Descriptive Statistics for Industry according to argument passed into function
    '''
    industry = industry.title()
    
    if industry == 'Financials':
        data = wrangle.finance

    elif industry == 'Utilities':
        data = wrangle.utils

    elif industry == 'Health Care':
        data = wrangle.health

    elif industry == 'Information Technology':
        data = wrangle.tech

    elif industry == 'Materials':
        data = wrangle.materials

    elif industry == 'Industrials':
        data = wrangle.industrials

    elif industry == 'Energy':
        data = wrangle.energy

    elif industry == 'Consumer Staples':
        data = wrangle.consumer_staple

    elif industry == 'Consumer Discretionary':
        data = wrangle.consumer_disc

    elif industry == 'Real Estate':
        data = wrangle.reits

    elif industry == 'Communication Services':
        data = wrangle.comm

    elif industry == 'All':
        industry = 'All Industries'
        data = wrangle.df

    log.info (f"Descriptive Statistics called for GICS Sector: {industry if industry != 'All' else 'All Sectors'} \n")    
    output = os.path.join(config.desc_path, f'{industry}.csv')
    desc = data.describe()
    
    with open(output, "w") as file:
        file.write(f"Descriptive Statistics for {industry} Companies\n")
        file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        file.write(str(desc))

