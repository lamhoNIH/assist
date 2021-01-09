import argparse
import json
import os
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *

def diagnostic_correlation(config_file, archive_path):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    Result(os.path.join(data_folder, archive_path), overwrite=False)
    config_path = os.path.join(data_folder, config_file)
    print("config_path: {}".format(config_path))

    with open(config_path) as json_data:
        config_json = json.load(json_data)

    computed_networks_df = pd.read_csv(os.path.join(data_folder, config_json["networks_of_coregulated_genes"]))
    comm_df1 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    comm_df2 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_agg1"]))

    comm_names = ['wgcna', 'louvain 1', 'louvain 2']
    comm_dfs = [computed_networks_df, comm_df1, comm_df2]
    deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    for i, cluster_df in enumerate(comm_dfs):
        cluster_DE_perc(deseq, cluster_df, 'louvain_label', comm_names[i])
        plot_sig_perc(cluster_df, 'louvain_label', comm_names[i], expression_meta_df)
        cluster_phenotype_corr(cluster_df, 'louvain_label', comm_names[i], expression_meta_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    
    diagnostic_correlation(args.config_file, args.archive_path)