import csv
import json
import sys
import tempfile

import extraction
from os import path, mkdir

# Value for prop_docker_mem = 28GB
def ade_entrypoint_v1(
    in_provided_networks,
    out_network_louvain_default, out_network_louvain_agg1,
    prop_docker_mem='30064771072',
    prop_docker_cpu='4', 
    prop_docker_volume_1='../..:/assist/data'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'config.json')

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'provided_networks': in_provided_networks
        },
        'outputs': {
            'network_louvain_default': out_network_louvain_default,
            'network_louvain_agg1': out_network_louvain_agg1
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    extraction.extract_modules(config_path)
    

if __name__ == '__main__':
    data_folder = '/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data'
    is_human = True
    
    if is_human:
        ade_entrypoint_v1(
            path.join(data_folder, 'Kapoor_TOM.csv'),
            path.join(data_folder, 'pipeline/human/module_extraction/ade_network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/human/module_extraction/ade_network_louvain_agg1.csv')
        )
    else:
         ade_entrypoint_v1(
            path.join(data_folder, 'pipeline/mouse/network_analysis/tom.csv'),
            path.join(data_folder, 'pipeline/mouse/module_extraction/ade_network_louvain_default.csv'),
            path.join(data_folder, 'pipeline/mouse/module_extraction/ade_network_louvain_agg1.csv')
        )