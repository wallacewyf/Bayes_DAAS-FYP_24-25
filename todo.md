1. extract past 20 years returns
2. extract past 20 years ESG scores
3. split into dataframe

    - 1, 2 and 3 are done
        - ROA / ROCE / ROE directly correlated, 
        - P/E ratio and ROE directly correlated
    
    - timeframe: 20 years

    - extracted scope:
        1. NASDAQ100 - based on BlackRock's iShares ETF fund
        2. S&P 500 - based on S&P Index
        3. MSCI World - based on SOURCE MSCI World Index
        4. STOXX 600: SOURCE STOXX EUROPE 600 UCITS ETF

        - cross-compare across BBG, Morningstar Sustainalytics?
    
4. run regression and correlation analysis
    - Time Series Regression Analysis
    - Correlation 
    - Relationship between ESG, E, S, G against Q ratio, YoY returns, ROA, ROE

5. extract results into folder in a presentable format


-- TODO LIST 
1. EXTRACT S&P 500 TOTAL ASSETS, REPORTED
2. FIGURE OUT HOW TO MOVE ALL DATA WRANGLING SCRIPTS INTO A DATA_WRANGLING FOLDER, AND LET CORR_ANALYSIS BE ABLE TO CALL IT FROM THE DATA_WRANGLING PATH
    (doesn't work - check with david)