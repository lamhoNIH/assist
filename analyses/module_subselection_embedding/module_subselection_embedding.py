import argparse
import json
import os
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *
from src.eda.subset_network import *

def module_subselection_approach1(data_folder, archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df):
    Result(os.path.join(data_folder, archive_path, "approach1", run_num))

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

    expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    for i, cluster_df in enumerate([comm_df1, subset_communities[0], subset_communities[1]]):
        cluster_DE_perc(cluster_df, 'louvain_label', subset_names[i] + " louvain", deseq)
        plot_sig_perc(cluster_df, 'louvain_label', subset_names[i], expression_meta_df)
        cluster_phenotype_corr(cluster_df, 'louvain_label', subset_names[i], expression_meta_df)

def module_subselection_approach2(data_folder, archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df):
    Result(os.path.join(data_folder, archive_path, "approach2", run_num))
    module4_tom = get_module_df(provided_networks_df, 4, comm_df1)

    # Original network with no cutoff
    scale_free_validate(provided_networks_df, 'whole network')

    output_path = Result.getPath()
    G0_n_4, module0_n_4_df = get_subnetwork1(0, 100, 0.015, provided_networks_df, comm_df1, module4_tom, plot_hist = False, subnetwork_file = os.path.join(Result.getPath(), 'module0_n_4_df.csv'))
    G1_n_4, module1_n_4_df = get_subnetwork1(1, 125, 0.01, provided_networks_df, comm_df1, module4_tom, plot_hist = False, subnetwork_file = os.path.join(Result.getPath(), 'module1_n_4_df.csv'))
    G2_n_4, module2_n_4_df = get_subnetwork1(2, 150, 0.01, provided_networks_df, comm_df1, module4_tom, plot_hist = False, subnetwork_file = os.path.join(Result.getPath(), 'module2_n_4_df.csv'))
    G3_n_4, module3_n_4_df = get_subnetwork1(3, 150, 0.02, provided_networks_df, comm_df1, module4_tom, plot_hist = False, subnetwork_file = os.path.join(Result.getPath(), 'module3_n_4_df.csv'))
    G4, module4_df = get_subnetwork2(250, 0.008, provided_networks_df, comm_df1, module4_tom, plot_hist = False, subnetwork_file = os.path.join(Result.getPath(), 'module4_df.csv'))

    subnetwork_dfs = []
    subnetwork_files = ['module0_n_4_df.csv','module1_n_4_df.csv', 'module2_n_4_df.csv', 'module3_n_4_df.csv', 'module4_df.csv']
    for file in subnetwork_files:
        df = pd.read_csv(os.path.join(Result.getPath(), file), index_col = 0)
        subnetwork_dfs.append(df)

    subnetwork_names = [file[:-4] for file in subnetwork_files]
    for i, subnetwork in enumerate(subnetwork_dfs):
        scale_free_validate(subnetwork, subnetwork_names[i])

    subnetwork_complete_dfs = []
    for subnetwork in subnetwork_dfs:
        subnetwork_complete_dfs.append(add_missing_genes(provided_networks_df, subnetwork))

    whole_and_subnetworks = [provided_networks_df] + subnetwork_complete_dfs
    plot_graph_distance(whole_and_subnetworks, ['all'] + subnetwork_names)

    # run louvain on subnetworks
    subset_communities = []
    for subset in subnetwork_dfs:
        subset_communities.append(run_louvain(subset, resolution = 0, n_aggregations = 0))

    for i, subset_com in enumerate(subset_communities):
        cluster_jaccard(comm_df1, subset_com, 'louvain_label', ['original', subnetwork_names[i]], cutout_nodes = True, top=3, y_max = 0.5)

    comparison_names = ['all vs ' + name for name in subnetwork_names]
    plot_cluster_nmi_comparison(comm_df1, subset_communities, 'louvain_label', comparison_names)

    all_network_names = ['all'] + subnetwork_names
    all_communities = [comm_df1] + subset_communities
    for i, cluster_df in enumerate(all_communities):
        cluster_DE_perc(cluster_df, 'louvain_label', all_network_names[i], deseq)
        plot_sig_perc(cluster_df, 'louvain_label', all_network_names[i], expression_meta_df)
        cluster_phenotype_corr(cluster_df, 'louvain_label', all_network_names[i], expression_meta_df)

def module_subselection(config_file, archive_path, run_num):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {} run_num: {}".format(config_file, data_folder, archive_path, run_num))
    config_path = os.path.join(data_folder, config_file)
    print("config_path: {}".format(config_path))

    with open(config_path) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)
    comm_df1 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    
    #module_subselection_approach1(data_folder, archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df)
    module_subselection_approach2(data_folder, archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    module_subselection(args.config_file, args.archive_path, args.run_num)