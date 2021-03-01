import argparse
import json
import os
import pandas as pd

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.embedding.fast_network_embedding import *
from src.eda.eda_functions import (plot_gene_cnt_each_cluster, get_closest_genes_jaccard,
                                   cluster_jaccard_v2, run_kmeans, cluster_DE_perc, plot_cluster_nmi_comparison_v3)

def run_embedding(config_file, archive_path, run_num):
    data_folder = Input.getPath()

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    network_df = pd.read_csv(os.path.join(data_folder, config_json["provided_networks"]), index_col=0)
    print("read provided_networks")

    Result(os.path.join(archive_path, run_num))
    output_path = Result.getPath()

    network_edgelist = adj_to_edgelist(network_df, output_dir = output_path)
    del network_edgelist # once it's saved, it can be deleted
    embed_params = config_json["embedding_params"]
    max_epoch = embed_params['max_epoch']
    learning_rate = embed_params['learning_rate']
    emb_df = network_embedding_fast(f'{output_path}/edgelist.txt',
                                    max_epoch=max_epoch, learning_rate=learning_rate,
                                    output_dir=output_path, name_spec=f'epoch={max_epoch}_alpha={learning_rate}')

    # run jaccard on the network modules vs embedding k means clusters
    comm_df = pd.read_csv(os.path.join(data_folder, config_json["network_louvain_default"]))
    kmeans = run_kmeans(emb_df)
    emb_name = f'epoch={max_epoch}_alpha={learning_rate}'
    cluster_jaccard_v2(comm_df, kmeans, 'louvain_label', 'kmean_label', ['Network', emb_name])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    run_embedding(args.config_file, args.archive_path, args.run_num)