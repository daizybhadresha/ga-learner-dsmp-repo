# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 
bank = pd.read_csv(path)

# Let's check which variable is categorical and which one is numerical so that you will get a basic idea about the features of the bank dataset.
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)


# --------------
## Filling the missing values
from scipy.stats import mode 
banks = bank.drop(columns = ['Loan_ID'])

print(banks.isnull().sum())

bank_mode = banks.mode()
print(bank_mode)

for x in banks.columns.values:
  banks[x]=banks[x].fillna(value=bank_mode[x].iloc[0])
   
print(banks.isnull().sum())


# --------------
## For basic idea of the average loan amount of a person - Loan Amount vs Gender

avg_loan_amount = pd.pivot_table(banks,index=['Gender','Married','Self_Employed'] , values='LoanAmount', aggfunc = np.mean)

print(avg_loan_amount)


# --------------
# The percentage of loan approved based on a person's employment type.
# Loan Approval vs Employment

loan_approved_se = banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')].shape[0]
loan_approved_nse = banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')].shape[0]

percentage_se = (loan_approved_se / 614) * 100
percentage_nse = (loan_approved_nse / 614) * 100


# --------------
# Transform the loan tenure from months to years
# company wants to find out those applicants with long loan amount term

loan_term = banks['Loan_Amount_Term'].apply(lambda m : m / 12)

big_loan_term = loan_term[loan_term>=25].shape[0]


# --------------
# Income/ Credit History vs Loan Amount
# the average income of an applicant and the average loan given to a person based on their income

loan_groupby = banks.groupby('Loan_Status')

loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]

mean_values = loan_groupby.mean()


