import argparse
import json
import os
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *
from src.eda.subset_network import *
from src.embedding.network_embedding import network_embedding

def module_subselection_approach1(archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df):
    Result(os.path.join(archive_path, "approach1", run_num))

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

def module_subselection_approach2(archive_path, run_num, config_json, provided_networks_df, comm_df, deseq, expression_meta_df):
    Result(os.path.join(archive_path, "approach2", run_num))

    # Original network with no cutoff
    scale_free_validate(provided_networks_df, 'whole network')

    subnetwork_path = Result.getPath()
    subnet_params = config_json["get_subnetwork_params"]
    n_clusters = len(subnet_params)
    
    subnetwork_Gs = []
    subnetwork_dfs = []
    subnetwork_names = []
    for p in subnet_params:
        print(f'non_deg_modules: {p["non_deg_modules"]}')
        G, module_df, subnetwork_name = get_subnetwork(p["deg_modules"], p["num_genes"], p["min_weight"], provided_networks_df, comm_df, deseq, non_deg_modules=p["non_deg_modules"], plot_hist = True, hist_dir = subnetwork_path, subnetwork_dir = subnetwork_path)
        subnetwork_Gs.append(G)
        subnetwork_dfs.append(module_df)
        subnetwork_names.append(subnetwork_name)

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
        cluster_jaccard(comm_df, subset_com, 'louvain_label', ['original', subnetwork_names[i]], cutout_nodes = True, top=3, y_max = 0.5)

    comparison_names = ['all vs ' + name for name in subnetwork_names]
    plot_cluster_nmi_comparison('all', comm_df, subset_communities, 'louvain_label', comparison_names)

    all_network_names = ['all'] + subnetwork_names
    all_communities = [comm_df] + subset_communities
    for i, cluster_df in enumerate(all_communities):
        cluster_DE_perc(cluster_df, 'louvain_label', all_network_names[i], deseq)
        if expression_meta_df is not None:
            plot_sig_perc(cluster_df, 'louvain_label', all_network_names[i], expression_meta_df)
            cluster_phenotype_corr(cluster_df, 'louvain_label', all_network_names[i], expression_meta_df)
        
    emb_list = []
    kmeans_list = []
    ep = config_json["embedding_params"]
    for i, G in enumerate(subnetwork_Gs):
        emb_df = network_embedding(G, ep["walk_length"], ep["num_walks"], ep["window"], subnetwork_path, subnetwork_names[i])
        emb_list.append(emb_df)
        kmeans_list.append(run_kmeans(emb_df, config_json["module_number"]))
        
    for i in range(1, n_clusters):
        cluster_jaccard(kmeans_list[0], kmeans_list[i], 'kmean_label', [subnetwork_names[0], subnetwork_names[i]], top = 3)

    network_comparison_names = [subnetwork_names[1] + f'vs {subnetwork_names[i]}' for i in range(n_clusters)]
    #plot_cluster_nmi_comparison(subnetwork_names[1], kmeans_list[1], kmeans_list, 'kmean_label', network_comparison_names)
    plot_cluster_nmi_comparison_v2([kmeans_list[0], kmeans_list[0], kmeans_list[1]], 
                                   [kmeans_list[1], kmeans_list[2], kmeans_list[2]], 
                                   [subnetwork_names[0], subnetwork_names[0], subnetwork_names[1]], 
                                   [subnetwork_names[1], subnetwork_names[2], subnetwork_names[2]], 
                                   'kmean_label')

    for i, kmeans in enumerate(kmeans_list):
        cluster_DE_perc(kmeans, 'kmean_label', subnetwork_names[i], deseq)
        if expression_meta_df is not None:
            plot_sig_perc(kmeans, 'kmean_label', subnetwork_names[i], expression_meta_df)
            cluster_phenotype_corr(kmeans, 'kmean_label', subnetwork_names[i], expression_meta_df)

    if "skip_kmeans_test" not in config_json or config_json["skip_kmeans_test"] is False:
        # 2x2 sets of parameters for embedding
        kmeans_list2 = []
        parameters = []
        etp = config_json["embedding_testing_params"]
        for length in etp["walk_length"]:
            for num_walk in etp["num_walks"]: # only use the first embedding to test different parameters based on the EDA
                emb_df = network_embedding(subnetwork_Gs[0], length, num_walk, etp["window"], Result.getPath(), subnetwork_names[0]) # use the network with 5k edges as a test (less computationally intensive)
                kmeans_list2.append(run_kmeans(emb_df, config_json["module_number"])) # run k means 
                parameters.append(f'length={length},num_walk={num_walk}') # add the parameter name to the parameters list
                
        for i in range(n_clusters-1):
            cluster_DE_perc(kmeans_list2[i], 'kmean_label', parameters[i], deseq)
            plot_sig_perc(kmeans_list2[i], 'kmean_label', parameters[i], expression_meta_df)
            cluster_phenotype_corr(kmeans_list2[i], 'kmean_label', parameters[i], expression_meta_df)
    
        kmeans_test = []
        emb = pd.read_csv(os.path.join(subnetwork_path, f'embedded_len{ep["walk_length"]}_walk{ep["num_walks"]}_module[{subnet_params[0]["deg_modules"][0]}]_n_[{subnet_params[0]["non_deg_modules"][0]}]_df.csv'), index_col = 0)
        n_list = config_json["kmeans_test_n_list"]
        for n in n_list:
            kmeans_test.append(run_kmeans(emb, n))
            
        for i in range(3):
            plot_sig_perc(kmeans_test[i], 'kmean_label', f'embedding 1 with {n_list[i]} clusters', expression_meta_df)
            cluster_phenotype_corr(kmeans_test[i], 'kmean_label', f'embedding 1 with {n_list[i]} clusters', expression_meta_df)

def module_subselection(config_file, archive_path, run_num):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {} run_num: {}".format(config_file, data_folder, archive_path, run_num))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    provided_networks_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col = 0)
    comm_df1 = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    if config_json["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    elif config_json["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    else:
        print(f'Unknown extension detected for {config_json["differentially_expressed_genes"]}')

    if ("skip_diagnostics" not in config_json) or (config_json["skip_diagnostics"] is False):
        expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    else:
        expression_meta_df = None
    
    #module_subselection_approach1(data_folder, archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df)
    module_subselection_approach2(archive_path, run_num, config_json, provided_networks_df, comm_df1, deseq, expression_meta_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    module_subselection(args.config_file, args.archive_path, args.run_num)