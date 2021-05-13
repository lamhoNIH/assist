import argparse
import json
import pandas as pd
from preproc.result import Result
from eda.eda_functions import *

def analyze_membership(config_file):
    print("config_file: {}".format(config_file))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    Result(config_json["parameters"]["plot_path"])

    gene_to_module_mapping_df = pd.read_csv(config_json["inputs"]["gene_to_module_mapping"])
    comm_df1 = pd.read_csv(config_json["inputs"]["network_louvain_default"])
    comm_df2 = pd.read_csv(config_json["inputs"]["network_louvain_agg1"])

    comm_dfs = [gene_to_module_mapping_df, comm_df1, comm_df2]
    comm_names = ['wgcna', 'louvain 1', 'louvain 2']
    for module_df, name in zip(comm_dfs, comm_names):
        plot_gene_cnt_each_cluster_v2(module_df, name)

    if ("skip_network_cluster_stability" not in config_json["parameters"]) or (json.loads(config_json["parameters"]["skip_network_cluster_stability"].lower()) is False):
        expression_meta_df = pd.read_csv(config_json["inputs"]["expression_with_metadata"], low_memory = False)
        cluster_pair_wgcna_n_com1, network_cluster_stability1 = network_cluster_stability(gene_to_module_mapping_df, comm_df1, 'cluster_id', expression_meta_df)
        for cluster in comm_df1.cluster_id.unique():
            plot_random_vs_actual_z(gene_to_module_mapping_df, comm_df1, cluster_pair_wgcna_n_com1[cluster], cluster, 'cluster_id', network_cluster_stability1, 'wgcna vs louvain 1', expression_meta_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    
    analyze_membership(args.config_file)