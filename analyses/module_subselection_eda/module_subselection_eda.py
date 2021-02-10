import argparse
import json
import os
import pandas as pd
import logging
import memory_profiler

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.subset_network import *

def module_subselection_approach2(logger, archive_path, run_num, data_folder, config_json, provided_networks_df, comm_df, deseq):
    Result(os.path.join(archive_path, "approach2", run_num))

    # Original network with no cutoff
    logger.debug("finished scale_free_validate on whole network")

    subnetwork_path = Result.getPath()
    subnet_params = config_json["subnetwork_params"]

    m1 = memory_profiler.memory_usage()
    for p in subnet_params:
        logger.debug(f'non_deg_modules: {p["non_deg_modules"]}')
        G, module_df, subnetwork_name = get_subnetwork(p["deg_modules"], p["num_genes"], p["min_weight"], provided_networks_df, comm_df, deseq, non_deg_modules=p["non_deg_modules"],
                                                       plot_hist = True, hist_dir = subnetwork_path, subnetwork_dir = None)

    m2 = memory_profiler.memory_usage()
    mem_diff = m2[0] - m1[0]
    logger.debug(f"finished loop for get_subnetwork m1: {m1} m2: {m2} mem_diff: {mem_diff}")

def module_subselection(config_file, archive_path, run_num):
    data_folder = Input.getPath()
    logfile = os.path.join(archive_path, "log.txt")
    logging.basicConfig(filename=logfile, filemode='w', format='%(asctime)s %(message)s')
    logger = logging.getLogger() 
    logger.setLevel(logging.DEBUG)
    logger.info("config_file: {} data_folder: {} archive_path: {} run_num: {}".format(config_file, data_folder, archive_path, run_num))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)
    logger.debug("read provided_networks")
    comm_df = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    logger.debug("read network_louvain_default")
    if config_json["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    elif config_json["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    else:
        logger.info(f'Unknown extension detected for {config_json["differentially_expressed_genes"]}')
        exit(2)
        
    logger.debug("read differentially_expressed_genes")
    module_subselection_approach2(logger, archive_path, run_num, data_folder, config_json, provided_networks_df, comm_df, deseq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    module_subselection(args.config_file, args.archive_path, args.run_num)