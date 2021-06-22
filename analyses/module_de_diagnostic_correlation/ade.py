import csv
import json
import sys
import tempfile

import diagnostic_correlation
from os import path, mkdir

# Value for prop_docker_mem = 10GB
def ade_entrypoint_v1(
    in_diagnostics, in_normalized_counts, in_gene_to_module_mapping, in_network_louvain_default, in_network_louvain_agg1, in_differentially_expressed_genes,
    out_expression_with_metadata,
    prop_plot_path,
    prop_docker_mem='10737418240',
    prop_docker_cpu='4', 
    prop_docker_volume_1='../..:/assist/data'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'module_de_diagnostic_correlation.json')

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'differentially_expressed_genes': in_differentially_expressed_genes,
            'diagnostics': in_diagnostics,
            'normalized_counts': in_normalized_counts,
            'gene_to_module_mapping': in_gene_to_module_mapping,
            'network_louvain_default': in_network_louvain_default,
            'network_louvain_agg1': in_network_louvain_agg1
        },
        'outputs': {
            'expression_with_metadata': out_expression_with_metadata
        },
        'parameters': {
            'plot_path': prop_plot_path
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    diagnostic_correlation.correlate_diagnostics(config_path)

if __name__ == '__main__':
    data_folder = '/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data'
    is_human = True
    
    if is_human:
        ade_entrypoint_v1(
            path.join(data_folder, 'Kapoor2019_coga.inia.detailed.pheno.04.12.17.csv'),
            path.join(data_folder, 'kapoor_expression_Apr5.txt'),
            path.join(data_folder, 'kapoor_wgcna_modules.csv'),
            path.join(data_folder, 'pipeline/human/module_extraction/network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/human/module_extraction/network_louvain_agg1.csv'),
            path.join(data_folder, 'deseq.alc.vs.control.age.rin.batch.gender.PMI.corrected.w.prot.coding.gene.name.xlsx'),
            path.join(data_folder, 'pipeline/mouse/module_de_diagnostic_correlation/artifacts'),
            'false'
        )
    else:
        ade_entrypoint_v1(
            path.join(data_folder, 'HDID_data/de_data.csv'),
            'NA',
            path.join(data_folder, 'pipeline/mouse/network_analysis/wgcna_modules.csv'),
            path.join(data_folder, 'pipeline/mouse/module_extraction/network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/mouse/module_extraction/network_louvain_agg1.csv'),
            path.join(data_folder, 'pipeline/mouse/module_membership_analysis/artifacts'),
            "true"
        )