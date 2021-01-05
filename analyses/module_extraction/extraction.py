import argparse
import json
import os
import shutil
import pandas as pd

from src.preproc.input import Input

Input('./Data')

from src.eda.eda_functions import *

def extract_modules(config_file, archive_path):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    Result(os.path.join(data_folder, archive_path))
    config_path = os.path.join(data_folder, config_file)
    shutil.copy(config_path, Result.getPath())

    with open(config_path) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)

    comm_df1 = run_louvain(provided_networks_df, 1, -1) # default setting
    comm_df2 = run_louvain(provided_networks_df, 1, 1)

    comm_df1.to_csv(os.path.join(Result.getPath(), config_json["network_louvain_default"]), index = 0)
    comm_df2.to_csv(os.path.join(Result.getPath(), config_json["network_louvain_agg1"]), index = 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    
    extract_modules(args.config_file, args.archive_path)