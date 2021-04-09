import argparse
import json
import os
import pandas as pd
import logging

#import sys
#sys.path.append('../../src')

from eda.eda_functions import *

def extract_modules(config_file):
    logfile = "log.txt"
    logging.basicConfig(filename=logfile, filemode='w', format='%(asctime)s %(message)s')
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    logger.info("config_file: {}".format(config_file))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(config_json["inputs"]["provided_networks"], index_col = 0)
    provided_networks_np = provided_networks_df.to_numpy()
    provided_networks_idx = provided_networks_df.index
    del provided_networks_df
    logger.debug("calling run_louvain2 for comm_df1")
    comm_df1 = run_louvain2(provided_networks_np, provided_networks_idx, 1, -1) # default setting
    comm_df1.to_csv(config_json["outputs"]["network_louvain_default"], index = 0)
    del comm_df1
    logger.debug("calling run_louvain2 for comm_df2")
    comm_df2 = run_louvain2(provided_networks_np, provided_networks_idx, 1, 1)
    comm_df2.to_csv(config_json["outputs"]["network_louvain_agg1"], index = 0)
    logger.debug("complete")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    
    extract_modules(args.config_file)