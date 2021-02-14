import argparse
import json
import os
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.subset_network import *

def module_subselection(config_file, archive_path, run_num):
    data_folder = Input.getPath()

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)
    print("read provided_networks")
    comm_df = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    print("read network_louvain_default")
    if config_json["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    elif config_json["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    else:
        print(f'Unknown extension detected for {config_json["differentially_expressed_genes"]}')
        exit(2)
        
    print("read differentially_expressed_genes")
    Result(os.path.join(archive_path, "approach2", run_num))

    subnetwork_path = Result.getPath()
    subnet_params = config_json["subnetwork_params"]

    for p in subnet_params:
        print(f'non_deg_modules: {p["non_deg_modules"]}')
        G, module_df, subnetwork_name = get_subnetwork(p["deg_modules"], p["num_genes"], p["min_weight"], provided_networks_df, comm_df, deseq, non_deg_modules=p["non_deg_modules"],
                                                       plot_hist = True, hist_dir = subnetwork_path, subnetwork_dir = None)

    print(f"finished loop for get_subnetwork")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    module_subselection(args.config_file, args.archive_path, args.run_num)