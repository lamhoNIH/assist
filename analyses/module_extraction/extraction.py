import pandas as pd
from src.eda.eda_functions import *

def extract_modules():
    tom_df = pd.read_csv('./Data/Kapoor_TOM.csv', index_col = 0)
    wgcna_modules = pd.read_csv('./Data/eda_derived/wgcna_modules.csv')
    comm_df1 = run_louvain(tom_df, 1, -1) # default setting
    comm_df2 = run_louvain(tom_df, 1, 1)
    comm_df1.to_csv('./Data/module_extraction/network_louvain_default.csv', index = 0)
    comm_df1.to_csv('./Data/module_extraction/network_louvain_agg1.csv', index = 0)

if __name__ == '__main__':
    extract_modules()