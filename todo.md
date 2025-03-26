Discussion with David:
- the norm in financial data is to log (but probably needs academic reference)
- remove top 5/10% of outliers to remove noise from the model 
    - but has to be specified in methodology since we're only assessing the impact of the framework on returns

Code:
    - multiple functions to try different models 
    - since reader might try different function 
    - basically there should be a function for each type of model i.e. gamma/gaussian glm, linear reg, linear reg w log
    - brief explanation on methodology section in paper
    - refer to previous functions in regression_OLD.py

Paper: 
    write about the trial and error process, but brief description on the error and a more detailed 
    one for the best fit one (evaluate according to p-value)

====================================================================================

Current thoughts (20250324):
- Write the code and functions for the below different type of regression:
    Linear Regression (DONE)
        - Basic Linear Regression (DONE)
        - Linear Regression with n-year shift (lag) (DONE)
        - Linear Regression with log-transformation on predictor variable Y (DONE)
        - Linear Regression with n-year shift and log-transformation (DONE)
        - Threshold Regression for the 3 categories above (DONE)

    Generalized Linear Models (GLM) (DONE)
        - Gaussian (DONE)
        - Inverse Gaussian (DONE)
        - Gamma (DONE) 
        - Tweedie (DONE) 

        - GLMs for each of the 5 categories above

        Note: Inverse Gaussian / Gamma / Tweedie requires user to adjust parameters since financial returns holds extreme outliers
              which makes the weights for GLMs to estimate not possible / feasible

- Descriptive Statistics (DONE)
    - make it so it parses any industry and drops it into the results/descriptive/ dir
    - doesn't have to be fancy, just make sure that it recognizes the industry

- Data Wrangling (DONE)
    - Decide on significance level of outliers (5/10% ?)
        - Currently set at 5% 

Maybe what the project is looking for is to provide a package with multiple statistical solutions to compute results
- Interpretation should be left to how the user interprets it rather than on personal interpretation
- We could add our interpretation but it doesn't have to be right?
- If it IS wrong then we can worry about that later