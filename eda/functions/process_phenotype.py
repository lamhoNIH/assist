import pandas as pd
from sys import platform

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

def get_expression_by_audit():
    expression_meta = pd.read_csv(prefix + '/Shared drives/NIAAA_ASSIST/Data/expression_meta.csv',
                                  low_memory = False)
    expression_meta['audit_category'] = 0
    expression_meta.loc[expression_meta['AUDIT'] < 25, 'audit_category'] = 'under 25'
    expression_meta.loc[expression_meta['AUDIT'].between(25, 50), 'audit_category'] = '25-50'
    expression_meta.loc[expression_meta['AUDIT'].between(50, 100), 'audit_category'] = '50-100'
    expression_meta.loc[expression_meta['AUDIT']> 100, 'audit_category'] = 'above 100'
    # remove rows without AUDIT labels
    audit_subset = expression_meta[expression_meta.audit_category != 0]
    return audit_subset