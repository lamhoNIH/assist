import csv
import json
import sys
import tempfile

import ml_and_cgi
from os import path, mkdir

# Value for prop_docker_mem = 18GB
def ade_entrypoint_v1(
    in_expression_with_metadata, in_embedding_file, in_differentially_expressed_genes, in_provided_networks,
    out_critical_genes,
    out_neighbor_genes,
    prop_plot_path,
    prop_skip_diagnostics,
    prop_top_n_coef,
    prop_models_to_find_cg,
    prop_aimed_cg_num,
    prop_aim_within_n,
    prop_top_n_critical_genes,
    prop_top_n_genes_for_comparison,
    prop_get_neighbor_genes,
    prop_docker_mem='19327352832',
    prop_docker_cpu='4', 
    prop_docker_volume_1='../..:/assist/data'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'network_embedding.json')

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'expression_with_metadata': in_expression_with_metadata,
            'differentially_expressed_genes': in_differentially_expressed_genes,
            'embedding_file': in_embedding_file,
            'provided_networks': in_provided_networks
        },
        'outputs': {
            'critical_genes': out_critical_genes,
            'neighbor_genes': out_neighbor_genes
        },
        'parameters': {
            'plot_path': prop_plot_path,
            'skip_diagnostics': prop_skip_diagnostics,
            'top_n_coef': float(prop_top_n_coef),
            'models_to_find_cg': json.loads(prop_models_to_find_cg),
            'aimed_cg_num': int(prop_aimed_cg_num),
            'aim_within_n': int(prop_aim_within_n),
            'top_n_critical_genes': int(prop_top_n_critical_genes),
            'top_n_genes_for_comparison': int(prop_top_n_genes_for_comparison),
            'get_neighbor_genes': prop_get_neighbor_genes
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    ml_and_cgi.ml_models(config_path)

if __name__ == '__main__':
    data_folder = '/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data'
    is_human = True

    if is_human:
        ade_entrypoint_v1(
            path.join(data_folder, 'pipeline/human/module_de_diagnostic_correlation/expression_meta.csv'),
            path.join(data_folder, 'pipeline/human/network_embedding/embedded_ggvec.csv'),
            path.join(data_folder, 'deseq.alc.vs.control.age.rin.batch.gender.PMI.corrected.w.prot.coding.gene.name.xlsx'),
            path.join(data_folder, 'Kapoor_TOM.csv'),
            path.join(data_folder, 'pipeline/human/ml_and_critical_gene_identifier/critical_gene_df.csv'),
            path.join(data_folder, 'pipeline/human/ml_and_critical_gene_identifier/neighbor_gene_df.csv'),
            path.join(data_folder, 'pipeline/human/ml_and_critical_gene_identifier/artifacts'),
            "false",
            "0.5",
            '["LR", "RF", "XGB"]',
            "850",
            "30",
            "10",
            "50",
            "true"
        )
    else:
        ade_entrypoint_v1(
            'NA',
            path.join(data_folder, 'HDID_data/de_data.csv'),
            'NA',
            path.join(data_folder, 'pipeline/mouse/network_embedding/embedding.csv'),
            path.join(data_folder, 'pipeline/mouse/ml_and_critical_gene_identifier/critical_gene_df.csv'),
            path.join(data_folder, 'pipeline/mouse/ml_and_critical_gene_identifier/neighbor_gene_df.csv'),
            path.join(data_folder, 'pipeline/mouse/ml_and_critical_gene_identifier/artifacts'),
            "true",
            "0.5",
            '["LR", "RF", "XGB"]',
            "500",
            "15",
            "10",
            'NA',
            'NA'
        )