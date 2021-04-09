import argparse
import json
import os
import pandas as pd

from preproc.result import Result

from eda.eda_functions import *

def correlate_diagnostics(config_file):
    print("config_file: {}".format(config_file))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    Result(config_json["parameters"]["plot_path"])
    
    gene_to_module_mapping_df = pd.read_csv(config_json["inputs"]["gene_to_module_mapping"])
    comm_df1 = pd.read_csv(config_json["inputs"]["network_louvain_default"])
    comm_df2 = pd.read_csv(config_json["inputs"]["network_louvain_agg1"])

    comm_names = ['wgcna', 'louvain 1', 'louvain 2']
    comm_dfs = [gene_to_module_mapping_df, comm_df1, comm_df2]
    split_tup = os.path.splitext(config_json["inputs"]["differentially_expressed_genes"])
    if config_json["inputs"]["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(config_json["inputs"]["differentially_expressed_genes"])
    elif config_json["inputs"]["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(config_json["inputs"]["differentially_expressed_genes"])
    else:
        print(f'Unknown extension {split_tup[1]} detected for {split_tup[0]}')
        # Added to work with ADE
        deseq = pd.read_csv(config_json["inputs"]["differentially_expressed_genes"])
    if ("skip_diagnostics" not in config_json["parameters"]) or (json.loads(config_json["parameters"]["skip_diagnostics"].lower()) is False):
        expression_meta = True
    else:
        expression_meta = False
    if expression_meta:
        expression_meta_df = pd.read_csv(config_json["inputs"]["expression_with_metadata"], low_memory = False)
    for i, cluster_df in enumerate(comm_dfs):
        cluster_DE_perc(cluster_df, 'louvain_label', comm_names[i], deseq)
        if expression_meta:
            plot_sig_perc(cluster_df, 'louvain_label', comm_names[i], expression_meta_df)
            cluster_phenotype_corr(cluster_df, 'louvain_label', comm_names[i], expression_meta_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    
    correlate_diagnostics(args.config_file)