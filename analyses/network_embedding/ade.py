import csv
import json
import sys
import tempfile

import network_embedding
from os import path, mkdir

# Value for prop_docker_mem = 16GB
def ade_entrypoint_v1(
    in_provided_networks, in_differentially_expressed_genes, in_network_louvain_default, 
    out_network_embedding,
    prop_plot_path,
    prop_max_epoch,
    prop_learning_rate,
    prop_docker_mem='17179869184',
    prop_docker_cpu='4', 
    prop_docker_volume_1='/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST:/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'network_embedding.json')

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'differentially_expressed_genes': in_differentially_expressed_genes,
            'expression_with_metadata': in_expression_with_metadata,
            'gene_to_module_mapping': in_gene_to_module_mapping,
            'network_louvain_default': in_network_louvain_default,
            'network_louvain_agg1': in_network_louvain_agg1
        },
        'parameters': {
            'plot_path': prop_plot_path,
            'skip_diagnostics': prop_skip_diagnostics
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    network_embedding.run_embedding(config_path)

if __name__ == '__main__':
    data_folder = '/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data'
    is_human = True
    
    if is_human:
        ade_entrypoint_v1(
            path.join(data_folder, 'Kapoor_TOM.csv'),
            path.join(data_folder, 'deseq.alc.vs.control.age.rin.batch.gender.PMI.corrected.w.prot.coding.gene.name.xlsx'),
            path.join(data_folder, 'pipeline/human/module_extraction/network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/human/network_embedding/embedded_ggvec.csv'),
            path.join(data_folder, 'pipeline/human/network_embedding/artifacts'),
            '100',
            '0.1'
        )
    else:
        ade_entrypoint_v1(
            path.join(data_folder, 'HDID_data/de_data.csv'),
            path.join(data_folder, 'pipeline/mouse/network_analysis/tom.csv'),
            path.join(data_folder, 'HDID_data/de_data.csv'),
            path.join(data_folder, 'pipeline/mouse/network_embedding/embedded_ggvec.csv'),
            path.join(data_folder, 'pipeline/mouse/network_embedding/artifacts'),
            "100",
            "0.1"
        )
