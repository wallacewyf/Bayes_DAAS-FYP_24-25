# packages
import sys, os, timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set path to access local packages
data_path = os.path.join(os.path.dirname(__file__), "data_wrangling")
sys.path.append(data_path)

# local packages
import config, wrangle

# initialize logger module 
log = config.logging

# Correlation Analysis

# Correlation between Index Components and Financial Returns
def index_corr(df_1, df_2):
    index_corr_arr = []
    log.info ('Running correlation analysis between index components and financial returns...')
    log.info ('--------------------------------------------------------------------')

    while sum(df_1.isnull().sum()) != sum(df_2.isnull().sum()):
        # TODO: dropna(axis = 1, inplace = True)
        df_1.dropna(inplace=True)
        df_2.dropna(inplace=True)

        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            log.info (f"{len(df_1.columns)} columns successfully aligned and extracted")
            total_year_avg = 0

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                log.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum
            
            index_corr_arr.append(total_year_avg/len(df_1.columns))

            break
        
        log.info ('')
        log.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        log.info ('--------------------------------------------------------------------')

        break

    else:
        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            log.info (f"{len(df_1.columns)} columns successfully aligned and extracted")

            total_year_avg = 0

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                log.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum

            index_corr_arr.append(total_year_avg/len(df_1.columns))

            break
        
        log.info ('')
        log.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        log.info ('--------------------------------------------------------------------')

    return index_corr_arr

# Correlation between GICS Industry Name and Financial Returns
def industry_corr(df_1, df_2):
    industry_corr_arr = []

    log.info ('Running correlation analysis between industries and financial returns...')
    log.info ('--------------------------------------------------------------------')

    while sum(df_1.isnull().sum()) != sum(df_2.isnull().sum()):
        df_1.dropna(inplace=True)
        df_2.dropna(inplace=True)

        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            log.info (f"{len(df_1.columns)} columns successfully aligned and extracted")

            df_1 = df_1.groupby(['GICS Industry Name']).mean()      # ESG Scores
            df_2 = df_2.groupby(['GICS Industry Name']).mean()      # Financial Ratio

            total_year_avg = 0

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                log.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum

            industry_corr_arr.append(total_year_avg/len(df_1.columns))

            break
        
        log.info ('')
        log.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        log.info ('--------------------------------------------------------------------')

        break

    else:
        scope = df_1.index.intersection(df_2.index)

        df_1 = df_1.reindex(scope)
        df_2 = df_2.reindex(scope)

        while len(df_1.columns) == len(df_2.columns):
            log.info (f"{len(df_1.columns)} columns successfully aligned and extracted")

            total_year_avg = 0

            df_1 = df_1.groupby(['GICS Industry Name']).mean()      # ESG Scores
            df_2 = df_2.groupby(['GICS Industry Name']).mean()      # Financial Ratio

            for x in range (len(df_1.columns)):
                avg_sum = 0 
                avg_sum += np.corrcoef(df_1.iloc[:,x], df_2.iloc[:,x])[0][1]

                log.info (f"Correlation for {df_1.columns[x]} = {avg_sum}")

                total_year_avg += avg_sum
            
            industry_corr_arr.append(total_year_avg/len(df_1.columns))

            break

        log.info ('')
        log.info (f"Total average correlation = {total_year_avg/len(df_1.columns)}")
        log.info ('')

    return industry_corr_arr

index_corr(df_1=
           wrangle.scores('msci', 'esg'),
           df_2=
           wrangle.returns('msci', 'roe'))

index = ['nasdaq', 'snp', 'stoxx', 'ftse', 'msci']
measures = ['roe', 'roa', 'mktcap', 'q']
esg = ['esg', 'e', 's', 'g']

def init_corr(idx):
    print ('Initializing correlation analysis...')

    if idx in index:
        for each_measure in measures:
            for each_esg in esg:
                log.info (f'Correlation by index initializing...')

                index_val = index_corr(df_1 = wrangle.scores(idx, each_esg), 
                                    df_2 = wrangle.returns(idx, each_measure))
                
                log.info (f'Correlation by index completed.')
                log.info (f'Correlation by industry initializing...')

                industry_val = industry_corr(df_1= wrangle.scores(idx, each_esg),
                                        df_2 = wrangle.returns(idx, each_measure))
                
                log.info (f"Correlation by industry completed.")
                log.info (f"Correlation analysis between {idx.upper()}'s {each_esg.upper()} scores and {each_measure.upper()} completed.")

        print ('Correlation Analysis completed.')
        print ('Check log file for results.')

    elif idx == 'all':
        for each_index in index:
            for each_measure in measures:
                for each_esg in esg:
                    log.info (f'Correlation by index initializing...')

                    index_val = index_corr(df_1 = wrangle.scores(each_index, each_esg), 
                                        df_2 = wrangle.returns(each_index, each_measure))
                    
                    log.info (f'Correlation by index completed.')
                    log.info ('')
                    log.info (f'Correlation by industry initializing...')

                    industry_val = industry_corr(df_1= wrangle.scores(each_index, each_esg),
                                            df_2 = wrangle.returns(each_index, each_measure))
                    
                    log.info (f"Correlation by industry completed.")
                    log.info ('')
                    log.info (f"Correlation analysis between {each_index.upper()}'s {each_esg.upper()} scores and {each_measure.upper()} completed.")

        print ('Correlation Analysis completed.')
        print ('Check log file for results.')

        return [index_val, industry_val]
