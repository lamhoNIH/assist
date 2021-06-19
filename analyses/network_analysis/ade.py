import csv
import json
import sys
import tempfile
from subprocess import Popen, PIPE

from os import path, mkdir

# Value for prop_docker_mem = 24GB
def ade_entrypoint_v1(
    in_normalized_counts,
    out_provided_networks, out_gene_to_module_mapping,
    prop_docker_mem='25769803776',
    prop_docker_cpu='4', 
    prop_docker_volume_1='../..:/assist/data'
):
    work_path = tempfile.mkdtemp()
    print(f'work_path: {work_path}')

    config_path = path.join(work_path, 'network_analysis.json')

    # Convert commas to tabs so the script can process it properly
    new_in_normalized_counts = path.join(work_path, 'in_normalized_counts.tsv')
    with open(in_normalized_counts, 'r') as f1, open(new_in_normalized_counts, 'w') as f2:
        reader = csv.reader(f1, delimiter=',')
        writer = csv.writer(f2, delimiter='\t')
        writer.writerows(reader)

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'normalized_counts': new_in_normalized_counts
        },
        'outputs': {
            'provided_networks': out_provided_networks,
            'gene_to_module_mapping': out_gene_to_module_mapping
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(' '.join(['Rscript', 'wgcna_codes.R', config_path]))

    process = Popen(['Rscript', 'wgcna_codes.R', config_path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    print(f'stdout: {stdout}')
    print(f'stderr: {stderr}', file=sys.stderr)
    print(f'exit_code: {exit_code}')
    if exit_code != 0:
        exit(exit_code)
    

if __name__ == '__main__':
    data_folder = '/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data'
    normalized_counts_but_with_comma_delim = path.join(
        tempfile.mkdtemp(),
        'normalized_counts_but_with_comma_delim.csv'
    )
    is_human = False
    input_normalized_counts = 'kapoor_expression.txt' if is_human else 'HDID_data/PFC_HDID_norm_exp.txt'
    with open(path.join(data_folder, input_normalized_counts), 'r') as f1, open(normalized_counts_but_with_comma_delim, 'w') as f2:
        reader = csv.reader(f1, delimiter='\t')
        writer = csv.writer(f2, delimiter=',')
        writer.writerows(reader)

    ade_entrypoint_v1(
        normalized_counts_but_with_comma_delim,
        path.join(data_folder, 'pipeline/mouse/network_analysis/tom.csv'),
        path.join(data_folder, 'pipeline/mouse/network_analysis/ade_wgcna_modules.csv')
    )