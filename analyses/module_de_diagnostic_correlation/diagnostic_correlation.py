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

    if 'diagnostics' in config_json["inputs"]:
        meta = pd.read_csv(config_json["inputs"]["diagnostics"])
        expression = pd.read_csv(config_json["inputs"]["normalized_counts"], sep = '\t', index_col = 0)
        expression_meta = pd.merge(expression.T, meta, left_index = True, right_on = 'IID')
        expression_meta.to_csv(os.path.join(config_json["outputs"]["expression_with_metadata"]), index = 0)

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
    for module_df, name in zip(comm_dfs, comm_names):
        cluster_DE_perc(module_df, name, deseq)
        if 'diagnostics' in config_json["inputs"]:
            plot_sig_perc(module_df, name, expression_meta_df)
            cluster_phenotype_corr(module_df, name, expression_meta_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    
    correlate_diagnostics(args.config_file)