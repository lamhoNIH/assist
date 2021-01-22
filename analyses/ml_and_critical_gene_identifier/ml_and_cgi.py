import argparse
import json
import os
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import seaborn as sns
import networkx as nx

from src.preproc.input import Input
from src.preproc.result import Result

Input('./Data')

from src.eda.subset_network import *
from src.eda.eda_functions import *
from src.eda.process_phenotype import *
from src.embedding.network_embedding import network_embedding
from src.models.feature_extraction import *
from src.models.ML_functions import *
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def ml_models(config_file, archive_path, run_num):
    data_folder = Input.getPath()
    print("config_file: {} data_folder: {} archive_path: {} run_num: {}".format(config_file, data_folder, archive_path, run_num))
    Result(os.path.join(data_folder, archive_path, run_num))
    config_path = os.path.join(data_folder, config_file)
    print("config_path: {}".format(config_path))

    with open(config_path) as json_data:
        config_json = json.load(json_data)
        
    embedding_path = os.path.join(data_folder, config_json["embedding_path"])
    embedding_prefix = config_json["embedding_prefix"]
    emb_list = []
    embedding_names = []
    for file in os.listdir(embedding_path):
        if file.startswith(embedding_prefix):
            emb = pd.read_csv(os.path.join(embedding_path, file), index_col = 0)
            emb_list.append(emb)
            emb_name = '_'.join(file.split('_')[-4:-1])
            embedding_names.append(emb_name)

    # process embedding to be ready for ML
    processed_emb_dfs = []
    deseq = pd.read_excel(os.path.join(data_folder, config_json["differentially_expressed_genes"]))
    for emb in emb_list:
        processed_emb_dfs.append(process_emb_for_ML(emb, deseq))

    model_weight_list = []
    for i, processed_df in enumerate(processed_emb_dfs):
        model_weight_list.append(run_ml(processed_df, emb_name = embedding_names[i], print_accuracy = True))

    top_dim_list = []
    for model_weights in model_weight_list:
        top_dim = plot_feature_importances(model_weights, top_n_coef = 0.5, print_num_dim = False, plot_heatmap = False, return_top_dim = True)
        top_dim_list.append(top_dim)
        
    for i, top_dim in enumerate(top_dim_list):
        jaccard_average(top_dim, embedding_names[i])

    critical_gene_sets = []
    critical_gene_dfs = []
    for i, processed_df in enumerate(processed_emb_dfs):
        gene_set = get_critical_gene_sets(processed_df, top_dim_list[i], max_dist = 2)
        critical_gene_sets.append(gene_set)
        critical_gene_dfs.append(get_critical_gene_df(gene_set, embedding_names[i], Result.getPath()))

    intersect_gene_list = []
    for i, critical_gene_df in enumerate(critical_gene_dfs):
        intersect_genes = jaccard_critical_genes(critical_gene_df, embedding_names[i])
        intersect_gene_list.append(intersect_genes)

    expression_meta_df = pd.read_csv(os.path.join(data_folder, config_json["expression_with_metadata"]), low_memory = False)
    gene_set_phenotype_corr(intersect_gene_list, embedding_names, expression_meta_df, 'intersect genes between 3 models')

    # critical_gene_sets2 is different from critical_gene_sets in that it only has # of nearby DEGs to the critical genes and is a complete list. 
    # critical_gene_sets2 only has gene IDs and only has the top 10 genes
    critical_gene_sets2 = []
    for i, critical_gene_df in enumerate(critical_gene_dfs):
        gene_set = plot_nearby_impact_num(critical_gene_df, embedding_names[i])
        critical_gene_sets2.append(gene_set)

    # Plot correlation of top critical genes (with most nearby impact genes) for each embedding
    gene_set_phenotype_corr(critical_gene_sets2, embedding_names, expression_meta_df, 'top 10 genes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    parser.add_argument("--archive_path", help="parent path to save results")
    parser.add_argument("--run_num", help="run number")
    args = parser.parse_args()
    ml_models(args.config_file, args.archive_path, args.run_num)