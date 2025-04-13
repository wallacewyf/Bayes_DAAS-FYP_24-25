# Project Title: Evaluating ESG (Environmental, Social, Governance) and its impact on corporate financial performance of Financial and Technology companies.

BSc Data Analytics and Actuarial Science

Bayes Business School (formerly Cass)

Authored by: Wallace Wong (200055530)

Supervisor: Dr David Smith

# Basic Configuration

This project uses Python 3.13, and statistical packages in Python such as Pandas, NumPy, Matplotlib, etc. that are available via the pip module. The below code, which would install the required statistical Python libraries, is recommended to be ran before running the GitHub repository to avoid errors. 

*pip install pandas, numpy, matplotlib, scikit-learn, seaborn, statsmodels, logger*

It is highly recommended to use the screens available on GitHub to have the data extracted in the same order on the Excel files, as the code extracts the data according to the order of the columns. If Refinitiv Eikon is not available, we recommend arranging the data in the same order as the above before running the scripts on GitHub.

The screens used within the Screener function on Eikon can also be found within the GitHub directory in the “eikon” folder, and the order of the columns of the downloaded Excel files can be found in the *data* directory.

Once the data is collected into Excel files, update their corresponding file locations in the config.py script.

i.e. 
    # Directory Path
    data_path = "C:/Users/xxx/Downloads/fyp_24/"

    # File Names
    scores = data_path + 'ESG Scores.xlsx'
    returns = data_path + 'Financial Returns.xlsx'

---

# Data Wrangling

The project also allows the wrangled dataset to be accessible and exported as comma-delimited files (csv), grouping by the GICS Sector’s classification. 

To access wrangled dataframes:
    Financials Sector: *wrangle.finance*
    Utilities Sector: *wrangle.utilities*


---

# Regression Models and Results 

The project uses the statsmodels package to run both the linear regression and generalized linear models for the aforementioned models in the Financials sector and provides the results in a file saved in a pre-specified directory in config.py's results_path

    - Examples can be found in examples.py
    - Models in file are to be found in main.py

Once the regression models are run, they will be saved in the results folder.

For example, when a Gaussian GLM is run between ESG and the Return on Assets (ROA) of firms in the Financials Sector: 

```bash
├───results
│   ├───descriptive
│   ├───glm
│   │   ├───basic_glm
│   │   │   └───Gaussian
│   │   │       └───Financials
│   │   │           └───ROA
│   │   │               └───ESG
```