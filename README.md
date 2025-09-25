# BRAC-Data-Analysis
BRAC Financial and Socio-Economic Analysis.

This project analyzes loan data from two BRAC product types, Dabi and Progati, from 2005 to 2014. The analysis explores how loan amounts relate to key economic and social indicators in Bangladesh, including GDP, inflation, unemployment, poverty, and the Human Development Index (HDI). The project includes data cleaning, visualization, and advanced statistical analysis using moderation and mediation models.

# Results
Descriptive Statistics
The all_data_combined dataset contains over 9.4 million observations.
This month Amount has a mean of approximately 6,491 and a standard deviation of about 1.57 million. The 
Cumulative Amount has a mean of approximately 469,265 and a standard deviation of about 20.25 million.
The Product type distribution shows a significantly higher count for 'consumer' loans compared to 'enterprise' loans. .
Key Findings from Plots
Dabi vs Progati Loans Over Years: The average loan amounts for both consumer and enterprise loans show significant volatility over the years.
Comparison of Normalized Loan Amounts and GDP: The normalized GDP shows a steady increase over time. The normalized consumer and enterprise loan amounts exhibit more fluctuations, suggesting their trends are not perfectly aligned with the overall economic growth trend.
OLS Regression Analysis Summary
The OLS regression results show very low R-squared values for the simple regressions, indicating that 
This month Amount alone has very little power to predict the macroeconomic variables. However, the moderation and mediation models provide more nuanced insights:
GDP Rate Regression: The simple model shows a very low R-squared of 0.000, and the Prob (F-statistic) is 0.000147, which indicates a statistically significant but weak relationship.
Inflation Rate Regression: The simple model also has a very low R-squared of 0.000, and the Prob (F-statistic) is 0.497, suggesting no significant linear relationship between This month Amount and Inflation Rate.
HDI Regression: The simple model for HDI shows a very low R-squared and Prob (F-statistic) of 0.271, indicating no significant relationship.
Moderation with Inflation Rate: The model with an interaction term between This month Amount and GDP Rate to predict Inflation Rate has an R-squared of 0.122, suggesting that the interaction term explains more of the variance than the simple model.
Large Condition Number: Many of the OLS regression summaries show a large Cond. No. (Condition Number), such as 1.58e+06. This warning suggests potential multicollinearity or other numerical problems, meaning some predictor variables are highly correlated and the model may be unstable.
