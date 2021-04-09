import csv
import json
import sys
import tempfile

import membership
from os import path, mkdir

# Value for prop_docker_mem = 16GB
def ade_entrypoint_v1(
    in_expression_with_metadata, in_gene_to_module_mapping, in_network_louvain_default, in_network_louvain_agg1,
    prop_plot_path,
    prop_skip_network_cluster_stability,
    prop_docker_mem='17179869184',
    prop_docker_cpu='4', 
    prop_docker_volume_1='/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST:/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'module_membership_analysis.json')

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'expression_with_metadata': in_expression_with_metadata,
            'gene_to_module_mapping': in_gene_to_module_mapping,
            'network_louvain_default': in_network_louvain_default,
            'network_louvain_agg1': in_network_louvain_agg1
        },
        'parameters': {
            'plot_path': prop_plot_path,
            'skip_network_cluster_stability': prop_skip_network_cluster_stability
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    membership.analyze_membership(config_path)

if __name__ == '__main__':
    data_folder = '/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data'
    is_human = True
    
    if is_human:
        ade_entrypoint_v1(
            path.join(data_folder, 'pipeline/human/network_analysis/wgcna_modules.csv'),
            path.join(data_folder, 'pipeline/human/network_analysis/expression_meta.csv'),
            path.join(data_folder, 'pipeline/human/module_extraction/network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/human/module_extraction/network_louvain_agg1.csv'),
            path.join(data_folder, 'pipeline/human/module_membership_analysis/artifacts'),
            'false'
        )
    else:
        ade_entrypoint_v1(
            path.join(data_folder, 'pipeline/mouse/network_analysis/wgcna_modules.csv'),
            'NA',
            path.join(data_folder, 'pipeline/mouse/module_extraction/network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/mouse/module_extraction/network_louvain_agg1.csv'),
            path.join(data_folder, 'pipeline/mouse/module_membership_analysis/artifacts'),
            "true"
        )