#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from os import makedirs, system
from os.path import exists
from shutil import copyfile
from sys import platform

# module name: [memory, tag]
module_map = {
    "network_analysis": ["24g", "0.1.0"], # 12 to 19 minutes on mouse
    "module_extraction": ["28g", "0.1.0"], # 9 to 15 minutes on human, 3 to 6 minutes on mouse
    "module_de_diagnostic_correlation": ["10g", "0.1.0"], # 10 to 17 minutes on human, 4 to 8 minutes on mouse
    "module_membership_analysis": ["16g", "0.1.0"], # < 1 minutes on human, 3 minutes on mouse
    "network_embedding": ["28g", "0.1.0"], # ~ 2 to 2.5 hrs on human, 1 hr on mouse
    "ml_and_critical_gene_identifier": ["18g", "0.1.0"] # 11 to 15 minutes on human, 6 minutes on mouse
}

script_dir = os.getcwd()
print(f'{script_dir}')
    
def run_module(data_dir, dataset, module):
    config_path = f"pipeline/{dataset}/{module}"
    
    if not exists(f"{data_dir}/{config_path}"):
        makedirs(f"{data_dir}/{config_path}")
    
    copyfile(f"{script_dir}/../config/{dataset}/{module}.json", f"{data_dir}/{config_path}/{module}.json")
    
    memory = module_map[module][0]
    tag = module_map[module][1]
    print(f"{datetime.now()}")
    system(f'docker run --rm -m {memory} -e config_file="data/{config_path}/{module}.json" -e archive_path="data/{config_path}" -v "{data_dir}":/assist/data assist/{module}:{tag}')
    print(f"{datetime.now()}")
    
def main():
    if len(sys.argv) < 3:
        print("python launch.py <module> <dataset> <optional absolute data path>")
        exit(2)
    
    module = sys.argv[1]
    if module not in module_map and module != "all":
    	print(f"module {module} is not found")
    	exit(2)
    
    dataset = sys.argv[2] # human or mouse
    
    if len(sys.argv) == 4:
    	data_dir = sys.argv[3]
    else:
    	# use default
    	prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
    	data_dir = f"{prefix}/Shared drives/NIAAA_ASSIST/Data"
    
    if module == "all":
        module_list = list(module_map.keys())
        if dataset == "human":
            module_list.remove("network_analysis")
    else:
        module_list = [module]
        
    for module in module_list:
        print(f"module: {module}")
        run_module(data_dir, dataset, module)

if __name__ == '__main__':
  main()