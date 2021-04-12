import csv
import json
import sys
import tempfile

import ml_and_cgi
from os import path, mkdir

# Value for prop_docker_mem = 20GB
def ade_entrypoint_v1(
    in_expression_with_metadata, in_differentially_expressed_genes, in_embedding_file, 
    out_critical_genes,
    prop_plot_path,
    prop_skip_diagnostics,
    prop_top_n_coef,
    prop_ratio,
    prop_docker_mem='21474836480',
    prop_docker_cpu='4', 
    prop_docker_volume_1='/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST:/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'network_embedding.json')

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'expression_with_metadata': in_expression_with_metadata,
            'differentially_expressed_genes': in_differentially_expressed_genes,
            'embedding_file': in_embedding_file
        },
        'outputs': {
            'critical_genes': out_critical_genes
        },
        'parameters': {
            'plot_path': prop_plot_path,
            'skip_diagnostics': prop_skip_diagnostics,
            'top_n_coef': prop_top_n_coef,
            'ratio': prop_ratio
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
            path.join(data_folder, 'pipeline/human/network_analysis/expression_meta.csv'),
            path.join(data_folder, 'deseq.alc.vs.control.age.rin.batch.gender.PMI.corrected.w.prot.coding.gene.name.xlsx'),
            path.join(data_folder, 'pipeline/human/network_embedding/embedded_ggvec.csv'),
            path.join(data_folder, 'pipeline/human/ml_and_critical_gene_identifier/critical_gene_df.csv'),
            path.join(data_folder, 'pipeline/human/network_embedding/artifacts'),
            "false",
            "0.5",
            "0.7"
        )
    else:
        ade_entrypoint_v1(
            'NA',
            path.join(data_folder, 'HDID_data/de_data.csv'),
            path.join(data_folder, 'pipeline/mouse/network_embedding/embedded_ggvec.csv'),
            path.join(data_folder, 'pipeline/mouse/ml_and_critical_gene_identifier/critical_gene_df.csv'),
            path.join(data_folder, 'pipeline/mouse/ml_and_critical_gene_identifier/artifacts'),
            "true",
            "0.5",
            "0.7"
        )