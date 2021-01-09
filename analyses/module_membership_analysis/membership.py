import argparse
import json
import os
import shutil
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *

def analyze_membership(config_file, archive_path):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    Result(os.path.join(data_folder, archive_path))
    config_path = os.path.join(data_folder, config_file)
    print("config_path: {}".format(config_path))
    shutil.copy(config_path, Result.getPath())

    with open(config_path) as json_data:
        config_json = json.load(json_data)

    computed_networks_df = pd.read_csv(os.path.join(data_folder, config_json["computed_networks"]))
    comm_df1 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    comm_df2 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_agg1"]))
    expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)

    comm_dfs = [computed_networks_df, comm_df1, comm_df2]
    comm_names = ['wgcna', 'louvain 1', 'louvain 2']
    plot_gene_cnt_each_cluster(comm_dfs, 'louvain_label', comm_names)
    
    cluster_pair_wgcna_n_com1, network_cluster_stability1 = network_cluster_stability(computed_networks_df, comm_df1, 'louvain_label', expression_meta_df)
    for cluster in comm_df1.louvain_label.unique():
        plot_random_vs_actual_z(computed_networks_df, comm_df1, cluster_pair_wgcna_n_com1[cluster], cluster, 'louvain_label', network_cluster_stability1, 'wgcna vs louvain 1', expression_meta_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    
    analyze_membership(args.config_file, args.archive_path)