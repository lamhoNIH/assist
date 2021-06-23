import argparse
import json
import os
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from preproc.result import Result
from eda.eda_functions import gene_phenotype_corr, plot_corr_kde
from eda.process_phenotype import *
from models.feature_extraction import *
from models.ML_functions import *

def ml_models(config_file):
    print("config_file: {}".format(config_file))

    with open(config_file) as json_data:
        config_json = json.load(json_data)

    Result(config_json["parameters"]["plot_path"])
    if 'get_neighbor_genes' in config_json['parameters']:
        if "provided_networks" not in config_json["inputs"]:
            print('To get neighbor genes, tom network must be provided.')
            exit(2)
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
    plot_ml_w_top_dim(processed_emb_df, top_dim)
    jaccard_average(top_dim, f'Important dim overlap within model repeats')

    cg_output = config_json['outputs']['critical_genes']
    aimed_cg_num = config_json['parameters']['aimed_cg_num']
    aim_within_n = config_json['parameters']['aim_within_n']
    models_to_find_cg = config_json['parameters']['models_to_find_cg']
    critical_gene_df = get_critical_gene_df(processed_emb_df, top_dim, deseq=deseq, output_path=cg_output,
                                            aimed_number=aimed_cg_num, within_n=aim_within_n, models=models_to_find_cg)
    intersect_genes = jaccard_critical_genes(critical_gene_df, f'Critical gene overlap between models')
    top_n_critical_genes = config_json['parameters']['top_n_critical_genes']
    critical_gene_set2 = plot_nearby_impact_num(critical_gene_df, emb_name, top = top_n_critical_genes)
    if ("skip_diagnostics" not in config_json['parameters']) or (json.loads(config_json['parameters']["skip_diagnostics"].lower()) is False):
        expression_meta_df = pd.read_csv(config_json["inputs"]["expression_with_metadata"], low_memory = False)
    else:
        expression_meta_df = None
    if 'get_neighbor_genes' in config_json['parameters']:
        if config_json['parameters']['get_neighbor_genes'] == True:
            tom_df = pd.read_csv(config_json["inputs"]["provided_networks"], index_col = 0)
            neighbor_gene_df = get_network_neighbor_genes(tom_df, deseq, len(critical_gene_df), within_n = aim_within_n)
            neighbor_gene_df.to_csv(config_json['outputs']['neighbor_genes'], index = 0)
    # Plot correlation of top critical genes with alcohol traits
        if expression_meta_df is not None:
            top_n_genes_for_comparison = config_json['parameters']['top_n_genes_for_comparison']
            cg_corr = gene_phenotype_corr(critical_gene_df.gene[:top_n_genes_for_comparison], expression_meta_df, 'Critical genes')
            deg_corr = gene_phenotype_corr(deseq.id[:top_n_genes_for_comparison], expression_meta_df, 'DEG')
            neighbor_corr = gene_phenotype_corr(critical_gene_df.gene[:top_n_genes_for_comparison], expression_meta_df, 'Neighbor genes')
            plot_corr_kde([cg_corr, deg_corr, neighbor_corr], ['CG', 'Neighbor', 'DEG'], 'CG, neighbor & DEG')
    else:
        if expression_meta_df is not None:
            top_n_genes_for_comparison = config_json['parameters']['top_n_genes_for_comparison']
            cg_corr = gene_phenotype_corr(critical_gene_df.gene[:top_n_genes_for_comparison], expression_meta_df, 'Critical genes')
            deg_corr = gene_phenotype_corr(deseq.id[:top_n_genes_for_comparison], expression_meta_df, 'DEG')
            plot_corr_kde([cg_corr, deg_corr], ['CG', 'DEG'], 'CG vs DEG')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to configuration file")
    args = parser.parse_args()
    ml_models(args.config_file)