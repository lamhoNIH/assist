import argparse
import json
import os
import pandas as pd
import logging
import memory_profiler

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.eda_functions import *
from src.eda.subset_network import *
from src.embedding.network_embedding import network_embedding

# Can be invoked after memory cleanup to conserve memory footprint
def get_expression_meta_df(data_folder, config_json):
    if ("skip_diagnostics" not in config_json) or (config_json["skip_diagnostics"] is False):
        expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    else:
        expression_meta_df = None
    return expression_meta_df

def module_subselection_approach1(archive_path, run_num, data_folder, config_json, provided_networks_df, comm_df, deseq):
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
        cluster_jaccard(comm_df, subset_com, 'louvain_label', ['original', subset_names[i]], cutout_nodes = False, top=3)
        cluster_jaccard(comm_df, subset_com, 'louvain_label', ['original', subset_names[i]], cutout_nodes = True, top=3, y_max = 0.05)

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
    plot_cluster_nmi_comparison(cluster1_name, comm_df, subset_communities, 'louvain_label', comparison_list)

    subset_names = config_json["louvain_list"]

    expression_meta_df = get_expression_meta_df(data_folder, config_json)
    for i, cluster_df in enumerate([comm_df, subset_communities[0], subset_communities[1]]):
        cluster_DE_perc(cluster_df, 'louvain_label', subset_names[i] + " louvain", deseq)
        if expression_meta_df is not None:
            plot_sig_perc(cluster_df, 'louvain_label', subset_names[i], expression_meta_df)
            cluster_phenotype_corr(cluster_df, 'louvain_label', subset_names[i], expression_meta_df)

def module_subselection_approach2(logger, archive_path, run_num, data_folder, config_json, provided_networks_df, comm_df, deseq):
    Result(os.path.join(archive_path, "approach2", run_num))

    # Original network with no cutoff
    scale_free_validate(provided_networks_df, 'whole network')
    logger.debug("finished scale_free_validate on whole network")

    subnetwork_path = Result.getPath()
    subnet_params = config_json["subnetwork_params"]
    n_clusters = len(subnet_params)
    logger.debug(f"n_clusters: {n_clusters}")
    
    subnetwork_Gs = []
    subnetwork_names = []
    subnetwork_nps = []
    subset_communities = []
    dc_distance_list = []
    ged_distance_list = [] 
    names = []
    whole_network_np = provided_networks_df.to_numpy()
    m1 = memory_profiler.memory_usage()
    for p in subnet_params:
        logger.debug(f'non_deg_modules: {p["non_deg_modules"]}')
        G, module_df, subnetwork_name = get_subnetwork(p["deg_modules"], p["num_genes"], p["min_weight"], provided_networks_df, comm_df, deseq, non_deg_modules=p["non_deg_modules"], plot_hist = True, hist_dir = subnetwork_path, subnetwork_dir = subnetwork_path)
        subnetwork_names.append(subnetwork_name)
        subnetwork_Gs.append(G)
        scale_free_validate(module_df, subnetwork_name)
        subset_communities.append(run_louvain(module_df, resolution = 0, n_aggregations = 0))
        subnetwork_complete = add_missing_genes(provided_networks_df, module_df)
        del module_df
        dc_distance, ged_distance = get_graph_distance(whole_network_np, subnetwork_complete.to_numpy())
        del subnetwork_complete
        dc_distance_list.append(dc_distance)
        ged_distance_list.append(ged_distance)
        names.append(f'all vs {subnetwork_name}')
    m2 = memory_profiler.memory_usage()
    mem_diff = m2[0] - m1[0]

    logger.debug(f"finished loop for scale_free_validate m1: {m1} m2: {m2} mem_diff: {mem_diff}")
    del provided_networks_df

    plot_graph_distances(dc_distance_list, ged_distance_list, names)
    del subnetwork_nps

    for i, subset_com in enumerate(subset_communities):
        cluster_jaccard(comm_df, subset_com, 'louvain_label', ['original', subnetwork_names[i]], cutout_nodes = True, top=3, y_max = 0.5)

    comparison_names = ['all vs ' + name for name in subnetwork_names]
    plot_cluster_nmi_comparison('all', comm_df, subset_communities, 'louvain_label', comparison_names)

    expression_meta_df = get_expression_meta_df(data_folder, config_json)
    all_network_names = ['all'] + subnetwork_names
    all_communities = [comm_df] + subset_communities
    for i, cluster_df in enumerate(all_communities):
        cluster_DE_perc(cluster_df, 'louvain_label', all_network_names[i], deseq)
        if expression_meta_df is not None:
            plot_sig_perc(cluster_df, 'louvain_label', all_network_names[i], expression_meta_df)
            cluster_phenotype_corr(cluster_df, 'louvain_label', all_network_names[i], expression_meta_df)
    #GBZ
    del comm_df
    for df in subset_communities:
        del df
    del subset_communities
    del all_communities
        
    emb_list = []
    kmeans_list = []
    ep = config_json["embedding_params"]
    for i, G in enumerate(subnetwork_Gs):
        emb_df = network_embedding(G, ep["walk_length"], ep["num_walks"], ep["window"], subnetwork_path, subnetwork_names[i])
        emb_list.append(emb_df)
        kmeans_list.append(run_kmeans(emb_df, config_json["module_number"]))
    #GBZ
    for emb_df in emb_list:
        del emb_df
    del emb_list
        
    logger.debug(f"n_clusters: {n_clusters} len(kmeans_list): {len(kmeans_list)} len(subnetwork_names): {len(subnetwork_names)}")
    for i in range(1, n_clusters):
        logger.debug(f"subnetwork_names[i]: {subnetwork_names[i]}")
        cluster_jaccard(kmeans_list[0], kmeans_list[i], 'kmean_label', [subnetwork_names[0], subnetwork_names[i]], top = 3)
        logger.debug("after cluster_jaccard")

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
        #GBZ
        del subnetwork_Gs
                
        for i in range(n_clusters-1):
            cluster_DE_perc(kmeans_list2[i], 'kmean_label', parameters[i], deseq)
            plot_sig_perc(kmeans_list2[i], 'kmean_label', parameters[i], expression_meta_df)
            cluster_phenotype_corr(kmeans_list2[i], 'kmean_label', parameters[i], expression_meta_df)
            
        #GBZ
        del deseq
    
        kmeans_test = []
        emb = pd.read_csv(os.path.join(subnetwork_path, f'embedded_len{ep["walk_length"]}_walk{ep["num_walks"]}_module[{subnet_params[0]["deg_modules"][0]}]_n_[{subnet_params[0]["non_deg_modules"][0]}]_df.csv'), index_col = 0)
        n_list = config_json["kmeans_test_n_list"]
        for n in n_list:
            kmeans_test.append(run_kmeans(emb, n))
            
        for i in range(3):
            plot_sig_perc(kmeans_test[i], 'kmean_label', f'embedding 1 with {n_list[i]} clusters', expression_meta_df)
            cluster_phenotype_corr(kmeans_test[i], 'kmean_label', f'embedding 1 with {n_list[i]} clusters', expression_meta_df)
            
        #GBZ
        del expression_meta_df

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
        logger.into(f'Unknown extension detected for {config_json["differentially_expressed_genes"]}')
        exit(2)
        
    logger.debug("read differentially_expressed_genes")

    #module_subselection_approach1(archive_path, run_num, data_folder, config_json, provided_networks_df, comm_df, deseq)
    module_subselection_approach2(logger, archive_path, run_num, data_folder, config_json, provided_networks_df, comm_df, deseq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    module_subselection(args.config_file, args.archive_path, args.run_num)