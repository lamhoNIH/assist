import argparse
import json
import pandas as pd
import memory_profiler
from preproc.result import Result
from embedding.fast_network_embedding import *
from eda.eda_functions import (plot_sig_perc, cluster_phenotype_corr, plot_corr_kde,
                               cluster_jaccard_v2, run_kmeans, cluster_DE_perc, cluster_nmi_v3,
                               plot_gene_cnt_each_cluster_v2)

def run_embedding(config_file):
    with open(config_file) as json_data:
        config_json = json.load(json_data)
    Result(config_json["parameters"]["plot_path"])
    
    print('before embedding', memory_profiler.memory_usage()[0])

    embed_params = config_json["parameters"]
    max_epoch = embed_params['max_epoch']
    learning_rate = embed_params['learning_rate']
    # Read tom file directly and then embed. Skip the saving edgelist step
    emb_df = network_embedding_fast(config_json["inputs"]["provided_networks"],
                                    max_epoch=max_epoch, learning_rate=learning_rate,
                                    output_path=config_json["outputs"]["embedding_path"])
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
    comm_df = pd.read_csv(config_json["inputs"]["chosen_module_assignment"])
    k = len(comm_df['cluster_id'].unique())
    kmeans = run_kmeans(emb_df, k)
    emb_name = f'epoch={max_epoch}_alpha={learning_rate}'
    cluster_DE_perc(comm_df, 'network', deseq)
    cluster_DE_perc(kmeans, f'{emb_name} embedding', deseq)

    # run jaccard on the network modules vs embedding k means clusters
    plot_gene_cnt_each_cluster_v2(comm_df, 'Network')
    plot_gene_cnt_each_cluster_v2(kmeans, emb_name)
    cluster_jaccard_v2(comm_df, kmeans, ['Network', emb_name])
    # run cluster gene and phenotype correlation
    if 'expression_with_metadata' in config_json["inputs"]:
        expression_meta_df = pd.read_csv(config_json["inputs"]["expression_with_metadata"], low_memory = False)
        network_cluster_corr = cluster_phenotype_corr(comm_df, 'network', expression_meta_df, output_corr_df=True)
        embedding_cluster_corr = cluster_phenotype_corr(kmeans, 'embedding', expression_meta_df, output_corr_df=True)
        plot_corr_kde([network_cluster_corr, embedding_cluster_corr], ['network', 'embedding'], 'network vs embedding')
    # run NMI
    print('NMI between network modules and embedding clusters is', cluster_nmi_v3(comm_df, kmeans))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    run_embedding(args.config_file)