import argparse
import json
import os
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *
from src.eda.subset_network import subset_network

def module_subselection(config_file, archive_path):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {}".format(config_file, data_folder, archive_path))
    Result(os.path.join(data_folder, archive_path), overwrite=False)
    config_path = os.path.join(data_folder, config_file)
    print("config_path: {}".format(config_path))

    with open(config_path) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)

    comm_df1 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    
    print(provided_networks_df.head(5))

    # Original network with no cutoff
    scale_free_validate(provided_networks_df, 'whole network')
    
    subset_networks = []
    for edge in config_json["subnet_edge_num_list"]:
        subset, G = subset_network(provided_networks_df, config_json["subnet_weight_min"], config_json["subnet_weight_max"], edge)
        subset_networks.append(subset) 
        scale_free_validate(subset, f'subnetwork with {edge} edges')

    subset_networks2 = []
    subset_G = []
    for weight in config_json["subnet2_weight_min_list"]:
        subset, G = subset_network(provided_networks_df, weight, config_json["subnet2_weight_max"])
        subset_networks2.append(subset)
        subset_G.append(G)
    plot_graph_distance(subset_networks2, config_json["subnet2_edge_num_list"])

    subset_communities = []
    for subset in subset_networks:
        subset_communities.append(run_louvain(subset))

    subset_names = config_json["subset_names"]
    for i, subset_com in enumerate(subset_communities):
        cluster_jaccard(comm_df1, subset_com, 'louvain_label', ['original', subset_names[i]], cutout_nodes = False, top=3)
        cluster_jaccard(comm_df1, subset_com, 'louvain_label', ['original', subset_names[i]], cutout_nodes = True, top=3, y_max = 0.05)

    # ask Yi-Pei how to distinguish the following two plots
    cluster1_name = config_json["subset_names"][0].split()[0]
    comparison_list = []
    for cn in config_json["subset_names"]:
        comparison_list.append(f'{cluster1_name} vs {cn.split()[0]}')
    plot_cluster_nmi_comparison(cluster1_name, subset_communities[0], subset_communities, 'louvain_label', comparison_list)

    cluster1_name = 'all'
    comparison_list = []
    for cn in config_json["subset_names"]:
        comparison_list.append(f'{cluster1_name} vs {cn.split()[0]}')
    plot_cluster_nmi_comparison(cluster1_name, comm_df1, subset_communities, 'louvain_label', comparison_list)

    subset_names = config_json["louvain_list"]
    deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    for i, cluster_df in enumerate([comm_df1, subset_communities[0], subset_communities[1]]):
        cluster_DE_perc(deseq, cluster_df, 'louvain_label', subset_names[i] + " louvain")
        plot_sig_perc(cluster_df, 'louvain_label', subset_names[i], expression_meta_df)
        cluster_phenotype_corr(cluster_df, 'louvain_label', subset_names[i], expression_meta_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="path to save results")
    args = parser.parse_args()
    module_subselection(args.config_file, args.archive_path)