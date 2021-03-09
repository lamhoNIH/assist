import argparse
import json
import os
import pandas as pd
import memory_profiler

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.embedding.fast_network_embedding import *
from src.eda.eda_functions import (plot_gene_cnt_each_cluster_v2, get_closest_genes_jaccard,
                                   cluster_jaccard_v2, run_kmeans, cluster_DE_perc, plot_cluster_nmi_comparison_v3)

def run_embedding(config_file, archive_path, run_num):
    data_folder = Input.getPath()

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    print('before embedding', memory_profiler.memory_usage()[0])
    Result(os.path.join(archive_path, run_num))
    output_path = Result.getPath()
    embed_params = config_json["embedding_params"]
    max_epoch = embed_params[0]['max_epoch']
    learning_rate = embed_params[0]['learning_rate']
    # Read tom file directly and then embed. Skip the saving edgelist step
    emb_df = network_embedding_fast(os.path.join(data_folder, config_json["provided_networks"]),
                                    max_epoch=max_epoch, learning_rate=learning_rate,
                                    output_dir=output_path, name_spec=f'epoch={max_epoch}_alpha={learning_rate}')
    print('after embedding', memory_profiler.memory_usage()[0])

    # Plot DEG % in each cluster
    if config_json["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    elif config_json["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    else:
        print(f'Unknown extension detected for {config_json["differentially_expressed_genes"]}')
        exit(2)
    # load data to run eda
    comm_df = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
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
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    run_embedding(args.config_file, args.archive_path, args.run_num)