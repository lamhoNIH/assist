import argparse
import json
import os
import shutil
import pandas as pd
import logging

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *

def extract_modules(config_file, archive_path):
    data_folder = Input.getPath()
    Result(archive_path)
    logfile = os.path.join(archive_path, "log.txt")
    logging.basicConfig(filename=logfile, filemode='w', format='%(asctime)s %(message)s')
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    logger.info("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    shutil.copy(config_file, Result.getPath())

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)
    provided_networks_np = provided_networks_df.to_numpy()
    provided_networks_idx = provided_networks_df.index
    del provided_networks_df
    logger.debug("calling run_louvain2 for comm_df1")
    comm_df1 = run_louvain2(provided_networks_np, provided_networks_idx, 1, -1) # default setting
    comm_df1.to_csv(os.path.join(Result.getPath(), config_json["network_louvain_default"]), index = 0)
    del comm_df1
    logger.debug("calling run_louvain2 for comm_df2")
    comm_df2 = run_louvain2(provided_networks_np, provided_networks_idx, 1, 1)
    comm_df2.to_csv(os.path.join(Result.getPath(), config_json["network_louvain_agg1"]), index = 0)
    logger.debug("complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    
    extract_modules(args.config_file, args.archive_path)