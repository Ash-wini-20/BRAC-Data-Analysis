import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import glob

# Load your datasets, replacing Google Drive paths with the new data folder path
loans_data = pd.read_csv('data/all_data_combined.csv')
gdp_data = pd.read_csv('data/bangladesh_gdp_rate.csv')
inflation_data = pd.read_csv('data/bangladesh_inflation_rate.csv')
poverty_data = pd.read_csv('data/bangladesh_poverty.csv')
hdi_data = pd.read_csv('data/bangladesh_development_index.csv')
unemployment_data = pd.read_csv('data/bangladesh_unemployment_rate.csv')

# Merge datasets
merged_data = loans_data.merge(gdp_data, on='Year').merge(inflation_data, on='Year').merge(poverty_data, on='Year', how='outer').merge(hdi_data, on='Year', how='outer').merge(unemployment_data, on='Year', how='outer')

# Normalize the loan amounts for visualization
scaler = StandardScaler()
merged_data[['Normalized Dabi Loans', 'Normalized Progati Loans']] = scaler.fit_transform(merged_data[['This month Amount', 'This month Amount']])

# Handle missing or infinite values
merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# ----------------- Moderation Analysis -----------------

# Example: Interaction term between loan amounts and Poverty Population
merged_data['Interaction_Term_Poverty'] = merged_data['This month Amount'] * merged_data['Poverty Population']

# Handle missing or infinite values and drop NaNs for the regression
merged_data.dropna(subset=['This month Amount', 'Poverty Population', 'Interaction_Term_Poverty', 'HDI'], inplace=True)

# Regression model with interaction term
X = merged_data[['This month Amount', 'Poverty Population', 'Interaction_Term_Poverty']]
y = merged_data['HDI']
X = sm.add_constant(X)
model_moderation_poverty = sm.OLS(y, X).fit()
print(model_moderation_poverty.summary())

# Example: Interaction term between loan amounts and Unemployment Rate
merged_data['Interaction_Term_Unemployment'] = merged_data['This month Amount'] * merged_data['Uneployment Rate']
merged_data.dropna(subset=['This month Amount', 'Uneployment Rate', 'Interaction_Term_Unemployment', 'GDP Rate'], inplace=True)
X = merged_data[['This month Amount', 'Uneployment Rate', 'Interaction_Term_Unemployment']]
y = merged_data['GDP Rate']
X = sm.add_constant(X)
model_moderation_unemployment = sm.OLS(y, X).fit()
print(model_moderation_unemployment.summary())

# ----------------- Mediation Analysis -----------------

# GDP as Mediator
# Step 1: Regress Loan Amounts on GDP Rate
X = merged_data[['This month Amount']]
y = merged_data['GDP Rate']
X = sm.add_constant(X)
model_step1_gdp = sm.OLS(y, X).fit()

# Step 2: Regress Loan Amounts on HDI
X = merged_data[['This month Amount']]
y = merged_data['HDI']
X = sm.add_constant(X)
model_step2_gdp = sm.OLS(y, X).fit()

# Step 3: Regress Loan Amounts and GDP Rate on HDI
X = merged_data[['This month Amount', 'GDP Rate']]
y = merged_data['HDI']
X = sm.add_constant(X)
model_step3_gdp = sm.OLS(y, X).fit()

print(model_step1_gdp.summary())
print(model_step2_gdp.summary())
print(model_step3_gdp.summary())

# Inflation rate as mediator
# Step 1: Regress Loan Amounts on Inflation Rate
X = merged_data[['This month Amount']]
y = merged_data['Inflation Rate']
X = sm.add_constant(X)
model_step1_inflation = sm.OLS(y, X).fit()

# Step 2: Regress Loan Amounts on GDP Rate
X = merged_data[['This month Amount']]
y = merged_data['GDP Rate']
X = sm.add_constant(X)
model_step2_inflation = sm.OLS(y, X).fit()

# Step 3: Regress Loan Amounts and Inflation Rate on GDP Rate
X = merged_data[['This month Amount', 'Inflation Rate']]
y = merged_data['GDP Rate']
X = sm.add_constant(X)
model_step3_inflation = sm.OLS(y, X).fit()

print(model_step1_inflation.summary())
print(model_step2_inflation.summary())
print(model_step3_inflation.summary())


# Poverty Population as mediator
# Step 1: Regress Loan Amounts on Poverty Population
X = merged_data[['This month Amount']]
y = merged_data['Poverty Population']
X = sm.add_constant(X)
model_step1_poverty = sm.OLS(y, X).fit()

# Step 2: Regress Loan Amounts on HDI
X = merged_data[['This month Amount']]
y = merged_data['HDI']
X = sm.add_constant(X)
model_step2_poverty = sm.OLS(y, X).fit()

# Step 3: Regress Loan Amounts and Poverty Population on HDI
X = merged_data[['This month Amount', 'Poverty Population']]
y = merged_data['HDI']
X = sm.add_constant(X)
model_step3_poverty = sm.OLS(y, X).fit()

print(model_step1_poverty.summary())
print(model_step2_poverty.summary())
print(model_step3_poverty.summary())

# HDI as mediator
# Step 1: Regress Loan Amounts on HDI
X = merged_data[['This month Amount']]
y = merged_data['HDI']
X = sm.add_constant(X)
model_step1_hdi = sm.OLS(y, X).fit()

# Step 2: Regress Loan Amounts on Unemployment Rate
X = merged_data[['This month Amount']]
y = merged_data['Uneployment Rate']
X = sm.add_constant(X)
model_step2_hdi = sm.OLS(y, X).fit()

# Step 3: Regress Loan Amounts and HDI on Unemployment Rate
X = merged_data[['This month Amount', 'HDI']]
y = merged_data['Uneployment Rate']
X = sm.add_constant(X)
model_step3_hdi = sm.OLS(y, X).fit()

print(model_step1_hdi.summary())
print(model_step2_hdi.summary())
print(model_step3_hdi.summary())

# Unemployment rate as mediator
# Step 1: Regress Loan Amounts on Unemployment Rate
X = merged_data[['This month Amount']]
y = merged_data['Uneployment Rate']
X = sm.add_constant(X)
model_step1_unemployment = sm.OLS(y, X).fit()

# Step 2: Regress Loan Amounts on GDP Rate
X = merged_data[['This month Amount']]
y = merged_data['GDP Rate']
X = sm.add_constant(X)
model_step2_unemployment = sm.OLS(y, X).fit()

# Step 3: Regress Loan Amounts and Unemployment Rate on GDP Rate
X = merged_data[['This month Amount', 'Uneployment Rate']]
y = merged_data['GDP Rate']
X = sm.add_constant(X)
model_step3_unemployment = sm.OLS(y, X).fit()

print(model_step1_unemployment.summary())
print(model_step2_unemployment.summary())
print(model_step3_unemployment.summary())

# ----------------- Visualization -----------------
# Ensure 'Year' is of type int and 'This month Amount' is numeric
loans_data['Year'] = pd.to_numeric(loans_data['Year'], errors='coerce')
loans_data['This month Amount'] = pd.to_numeric(loans_data['This month Amount'], errors='coerce')

# Drop rows with NaN values in 'Year' or 'This month Amount'
loans_data.dropna(subset=['Year', 'This month Amount'], inplace=True)

# Aggregate data by year and product type
loan_comparison = loans_data.groupby(['Year', 'Product type']).agg({'This month Amount': 'mean'}).reset_index()

# Plotting Dabi vs Progati Loans Over Years
plt.figure(figsize=(14, 7))
sns.lineplot(data=loan_comparison, x='Year', y='This month Amount', hue='Product type')
plt.title('Dabi vs Progati Loans Over Years')
plt.xlabel('Year')
plt.ylabel('This month Amount')
plt.legend(title='Product type')
plt.grid(True)
plt.show()

# Group by Year and Month
monthly_data = loans_data.groupby(['Year', 'Month', 'Product type']).agg({'This month Amount': 'mean'}).reset_index()

# Plotting Monthly Loan Amounts Over Years
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_data, x='Month', y='This month Amount', hue='Year')
plt.title('Monthly Loan Amounts Over Years')
plt.show()

# Bar plot of 'Product type' distribution
loans_data['Product type'].value_counts().plot(kind='bar', color='lightgreen')
plt.xlabel('Product Type')
plt.ylabel('Count')
plt.title('Distribution of Product Types')
plt.show()

# Plotting normalized data
# Load the GDP data
gdp_data = pd.read_csv('data/bangladesh_gdp.csv') # Assuming bangladesh_gdp.csv exists

# Ensure 'Year' is of type int and 'GDP' is numeric
gdp_data['Year'] = pd.to_numeric(gdp_data['Year'], errors='coerce')
gdp_data['GDP'] = pd.to_numeric(gdp_data['GDP'], errors='coerce')

# Normalize the loan amounts and GDP values for comparison
loan_comparison['Normalized Loans'] = loan_comparison['This month Amount'] / loan_comparison['This month Amount'].max()
gdp_data['Normalized GDP'] = gdp_data['GDP'] / gdp_data['GDP'].max()

# Merge the loan data with the GDP data
merged_data_gdp = pd.merge(loan_comparison, gdp_data, on='Year')

# Plotting the normalized data
plt.figure(figsize=(14, 7))
sns.lineplot(data=merged_data_gdp, x='Year', y='Normalized Loans', hue='Product type', marker='o')
sns.lineplot(data=merged_data_gdp, x='Year', y='Normalized GDP', color='black', label='Normalized GDP', marker='x')
plt.title('Comparison of Normalized Loan Amounts and GDP Over Years')
plt.xlabel('Year')
plt.ylabel('Normalized Values')
plt.legend(title='Legend')
plt.grid(True)
plt.show()
