import argparse
import os
import pandas as pd

from src.preproc.input import Input

Input('./Data')

from src.eda.eda_functions import *

def extract_modules(archive_path):
    data_folder = Input.getPath()
    tom_df = pd.read_csv(os.path.join(data_folder, 'Kapoor_TOM.csv'), index_col = 0)
    #wgcna_modules = pd.read_csv(os.path.join(data_folder, 'network_analysis/wgcna_modules.csv'))
    comm_df1 = run_louvain(tom_df, 1, -1) # default setting
    comm_df2 = run_louvain(tom_df, 1, 1)
    output_path = os.path.join(data_folder, archive_path)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    comm_df1.to_csv(os.path.join(output_path, 'network_louvain_default.csv'), index = 0)
    comm_df2.to_csv(os.path.join(output_path, 'network_louvain_agg1.csv'), index = 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    
    extract_modules(args.archive_path)