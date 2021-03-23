import csv
import json
import shutil
import sys
import tempfile
import preproc


def ade_entrypoint_v1(
    in_diagnostics, in_normalized_counts,
    out_expression_with_metadata, out_gene_to_module_mapping,
    prop_skip_tom='true', skip_preproc='false',
    prop_docker_mem='10737418240', prop_docker_cpu='4', prop_docker_volume_1='/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST:/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST'
):
    work_path = tempfile.mkdtemp()

    config_path = path.join(work_path, 'config.json')
    archive_path = path.join(work_path, 'output')

    # Convert commas to tabs so the script can process it properly
    new_in_normalized_counts = path.join(work_path, 'in_normalized_counts.tsv')
    with open(in_normalized_counts, 'rU') as f1, open(new_in_normalized_counts, 'w') as f2:
        reader = csv.reader(f1, delimiter=',')
        writer = csv.writer(f2, delimiter='\t')
        writer.writerows(reader)

    # Generate and write out JSON
    # CONFIG.JSON EXAMPLE: G:\Shared drives\NIAAA_ASSIST\Data\pipeline\human\network_analysis\config.json
    config = {
        'inputs': {
            'diagnostics': in_diagnostics,
            'normalized_counts': in_normalized_counts
        },
        'outputs': {
            'expression_with_metadata': out_expression_with_metadata,
            'gene_to_module_mapping': out_gene_to_module_mapping
        },
        'parameters': {
            'skip_tom': prop_skip_tom,
            'skip_preproc': skip_preproc
        }
    }

    with open(config_path, 'w') as f:
        json.dump(config, f)

    preproc(config_path, archive_path)
    process = Popen(['Rscript', 'wgcna_codes.R', config_path, archive_path], stdout=PIPE)
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    print(f'{stdout}')
    print(f'{stderr}', file=sys.stderr)
    if exit_code != 0:
        exit(exit_code)
    

if __name__ == '__main__':
    ade_entrypoint_v1(
        'G:\\Shared drives\\NIAAA_ASSIST\\Data\\kapoor_expression.txt',
        'G:\\Shared drives\\NIAAA_ASSIST\\Data\\kapoor2019_coga.inia.detailed.pheno.04.12.17.csv',
        'G:\\Shared drives\\NIAAA_ASSIST\\Data\\pipeline\\human\\network_analysis\\KF_test\\expression_meta.csv',
        'G:\\Shared drives\\NIAAA_ASSIST\\Data\\pipeline\\human\\network_analysis\\KF_test\\wgcna_modules.csv'
    )