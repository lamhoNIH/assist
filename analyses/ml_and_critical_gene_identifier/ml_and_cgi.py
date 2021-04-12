import argparse
import json
import os
import pandas as pd
pd.set_option('mode.chained_assignment', None)

from preproc.result import Result

from eda.eda_functions import gene_set_phenotype_corr
from eda.process_phenotype import *
from models.feature_extraction import *
from models.ML_functions import *

def ml_models(config_file):
    print("config_file: {}".format(config_file))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    Result(config_json["parameters"]["plot_path"])

    emb_df = pd.read_csv(config_json["inputs"]["embedding_file"], index_col = 0)
    emb_name = '_'.join(os.path.basename(config_json["inputs"]["embedding_file"]).split('_')[2:])[:-4]
    
    if config_json["inputs"]["differentially_expressed_genes"].endswith(".xlsx"):
        deseq = pd.read_excel(config_json["inputs"]["differentially_expressed_genes"])
    elif config_json["inputs"]["differentially_expressed_genes"].endswith(".csv"):
        deseq = pd.read_csv(config_json["inputs"]["differentially_expressed_genes"])
    else:
        print(f'Unknown extension detected for {config_json["inputs"]["differentially_expressed_genes"]}')
        # added for ADE integration
        deseq = pd.read_csv(config_json["inputs"]["differentially_expressed_genes"])
    #deseq['abs_log2FC'] = abs(deseq['log2FoldChange'])
    # process embedding to be ready for ML
    processed_emb_df = process_emb_for_ML(emb_df, deseq)

    model_weights = run_ml(processed_emb_df, emb_name=emb_name, print_accuracy=True)
    top_dim = plot_feature_importances(model_weights, top_n_coef=float(config_json["parameters"]["top_n_coef"]), print_num_dim=False, plot_heatmap=False,
                                       return_top_dim=True)
    plot_random_feature_importance(model_weights, top_dim, emb_name)
    jaccard_average(top_dim, f'Important dim overlap between models')
    ratio = float(config_json['parameters']['ratio'])
    gene_set = get_critical_gene_sets(processed_emb_df, top_dim, deseq, ratio = ratio)
    is_0_cnt = 0
    for i in range(len(gene_set)):
        if len(gene_set[i][0]) == 0:
            is_0_cnt += 1
    if is_0_cnt > 0:
        print('Critical gene identification INCOMPLETE')
        print(f'{is_0_cnt} out of 9 models identified 0 critical genes. Try increasing ratio')
        exit(1)
    critical_gene_df = get_critical_gene_df(gene_set, emb_name, Result.getPath())
    intersect_genes = jaccard_critical_genes(critical_gene_df, f'Critical gene overlap between models')

    if ("skip_diagnostics" not in config_json['parameters']) or (json.loads(config_json['parameters']["skip_diagnostics"].lower()) is False):
        expression_meta_df = pd.read_csv(config_json["inputs"]["expression_with_metadata"], low_memory = False)
    else:
        expression_meta_df = None

    if expression_meta_df is not None:
        gene_set_phenotype_corr([intersect_genes], [emb_name], expression_meta_df, 'common genes across the 3 models')

    # critical_gene_sets2 is different from critical_gene_sets in that it only has # of nearby DEGs to the critical genes and is a complete list. 
    # critical_gene_sets2 only has gene IDs and only has the top 10 genes
    critical_gene_sets2 = plot_nearby_impact_num(critical_gene_df, emb_name)

    # Plot correlation of top critical genes (with most nearby impact genes) for each embedding
    if expression_meta_df is not None:
        gene_set_phenotype_corr([critical_gene_sets2], [emb_name], expression_meta_df, 'Top 10 critical genes')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    ml_models(args.config_file)