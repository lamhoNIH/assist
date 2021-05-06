import argparse
import json
import os
import pandas as pd
import memory_profiler

from preproc.result import Result

from embedding.fast_network_embedding import *
from eda.eda_functions import (plot_gene_cnt_each_cluster_v2, get_closest_genes_jaccard,
                                   cluster_jaccard_v2, run_kmeans, cluster_DE_perc, plot_cluster_nmi_comparison_v3)

def run_embedding(config_file):

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    Result(config_json["parameters"]["plot_path"])
    
    print('before embedding', memory_profiler.memory_usage()[0])

    embed_params = config_json["parameters"]
    max_epoch = int(embed_params['max_epoch'])
    learning_rate = float(embed_params['learning_rate'])
    # Read tom file directly and then embed. Skip the saving edgelist step
    emb_df = network_embedding_fast(config_json["inputs"]["provided_networks"],
                                    max_epoch=max_epoch, learning_rate=learning_rate,
                                    output_path=config_json["outputs"]["network_path"])
    print('after embedding', memory_profiler.memory_usage()[0])

    # Plot DEG % in each cluster
    if config_json["inputs"]["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(config_json["inputs"]["differentially_expressed_genes"])
    elif config_json["inputs"]["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(config_json["inputs"]["differentially_expressed_genes"])
    else:
        print(f'Unknown extension detected for {config_json["inputs"]["differentially_expressed_genes"]}')
        # Added for ADE integration
        deseq = pd.read_csv(config_json["inputs"]["differentially_expressed_genes"])
    # load data to run eda
    comm_df = pd.read_csv(config_json["inputs"]["network_louvain_default"])
    k = len(comm_df['louvain_label'].unique())
    kmeans = run_kmeans(emb_df, k)
    emb_name = f'epoch={max_epoch}_alpha={learning_rate}'
    cluster_DE_perc(comm_df, 'louvain_label', 'network', deseq)
    cluster_DE_perc(kmeans, 'kmean_label', f'{emb_name} embedding', deseq)

    # run jaccard on the network modules vs embedding k means clusters
    plot_gene_cnt_each_cluster_v2(comm_df, 'louvain_label', 'Network', '_network')
    plot_gene_cnt_each_cluster_v2(kmeans, 'kmean_label', emb_name, '_embedding')
    cluster_jaccard_v2(comm_df, kmeans, 'louvain_label', 'kmean_label', ['Network', emb_name])
    # run NMI
    network_comparison_name = ['Network' + f' vs {emb_name} embedding']
    plot_cluster_nmi_comparison_v3('Network', cluster1 = comm_df, cluster1_column = 'louvain_label',
                                   cluster2_list= [kmeans], cluster2_column = 'kmean_label', comparison_names = network_comparison_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    run_embedding(args.config_file)