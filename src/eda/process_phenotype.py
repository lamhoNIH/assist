import pandas as pd
from sys import platform

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

def get_expression_by_audit(expression_meta_df):
    expression_meta_df['audit_category'] = 0
    expression_meta_df.loc[expression_meta_df['AUDIT'] < 25, 'audit_category'] = 'under 25'
    expression_meta_df.loc[expression_meta_df['AUDIT'].between(25, 50), 'audit_category'] = '25-50'
    expression_meta_df.loc[expression_meta_df['AUDIT'].between(50, 100), 'audit_category'] = '50-100'
    expression_meta_df.loc[expression_meta_df['AUDIT']> 100, 'audit_category'] = 'above 100'
    # remove rows without AUDIT labels
    audit_subset = expression_meta_df[expression_meta_df.audit_category != 0]
    return audit_subset

def get_liver_class(expression_meta_df):
    expression_meta_w_liver_class = expression_meta_df[~expression_meta_df['Liver_class'].isna()]
    return expression_meta_w_liver_class

def get_expression_by_alcohol_perday(expression_meta_df):
    expression_meta_df['alcohol_intake_category'] = 0
    expression_meta_df.loc[expression_meta_df['alcohol_intake_gmsperday'] < 50, 'alcohol_intake_category'] = 'under 50'
    expression_meta_df.loc[expression_meta_df['alcohol_intake_gmsperday'].between(50, 100), 'alcohol_intake_category'] = '50-100'
    expression_meta_df.loc[expression_meta_df['alcohol_intake_gmsperday'].between(100, 300), 'alcohol_intake_category'] = '100-300'
    expression_meta_df.loc[expression_meta_df['alcohol_intake_gmsperday']> 300, 'alcohol_intake_category'] = 'above 300'
    alc_intake_subset = expression_meta_df[expression_meta_df.alcohol_intake_category != 0]
    return alc_intake_subset

def get_expression_by_drinking_yrs(expression_meta_df):
    expression_meta_df['drinking_yrs_category'] = 0
    expression_meta_df.loc[expression_meta_df['Total_drinking_yrs'] < 20, 'drinking_yrs_category'] = 'under 20'
    expression_meta_df.loc[expression_meta_df['Total_drinking_yrs'].between(20, 30), 'drinking_yrs_category'] = '20-30'
    expression_meta_df.loc[expression_meta_df['Total_drinking_yrs'].between(30, 40), 'drinking_yrs_category'] = '30-40'
    expression_meta_df.loc[expression_meta_df['Total_drinking_yrs']> 40, 'drinking_yrs_category'] = 'above 40'
    drink_yrs_subset = expression_meta_df[expression_meta_df.drinking_yrs_category != 0]
    return drink_yrs_subset

def get_smoke_freq(expression_meta_df):
    expression_meta_w_smoke_freq = expression_meta_df[~expression_meta_df['Smoking_frequency'].isna()]
    expression_meta_w_smoke_freq = expression_meta_w_smoke_freq[~expression_meta_w_smoke_freq['Smoking_frequency'].str.contains('Not reported')]
    expression_meta_w_smoke_freq['Smoking_frequency'] = expression_meta_w_smoke_freq['Smoking_frequency'].apply(lambda x: x.split('-')[1].strip())
    return expression_meta_w_smoke_freq