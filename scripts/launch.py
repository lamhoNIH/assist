#!/usr/bin/env python3

import os
import sys
from datetime import datetime
from os import makedirs, system
from os.path import exists
from shutil import copyfile
from sys import platform

script_dir = os.getcwd()
print(f'{script_dir}')

# module name: [memory, tag]
module_map = {
    "network_analysis": ["24g", "0.1.0"], # ~3 to 4 minutes on human, 12 to 19 minutes on mouse
    "module_extraction": ["28g", "0.1.0"], # ~15 minutes on human, 6 minutes on mouse
    "module_membership_analysis": ["16g", "0.1.0"], # ~14 minutes on human, 3 minutes on mouse
    "module_de_diagnostic_correlation": ["10g", "0.1.0"], # ~17 minutes on human, 8 minutes on mouse
    "network_embedding": ["28g", "0.1.0"], # ~ 2.5 hrs on human, 1 hr on mouse
	"ml_and_critical_gene_identifier": ["18g", "0.1.0"] # ~11 to 15 minutes on human, 6 minutes on mouse
}

if len(sys.argv) < 3:
    print("python launch.py <module> <dataset> <optional data path>")
    exit(2)

module = sys.argv[1]
if module not in module_map:
	print(f"module {module} is not found")
	exit(2)

dataset = sys.argv[2] # human or mouse

if len(sys.argv) == 4:
	data = sys.argv[3]
else:
	# use default
	prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
	data = f"{prefix}/Shared drives/NIAAA_ASSIST/Data"

config_path = f"pipeline/{dataset}/{module}"

if not exists(f"{data}/{config_path}"):
    makedirs(f"{data}/{config_path}")

copyfile(f"{script_dir}/../config/{dataset}/{module}.json", f"{data}/{config_path}/{module}.json")

memory = module_map[module][0]
tag = module_map[module][1]
print(f"{datetime.now()}")
system(f'docker run --rm -m {memory} -e config_file="data/{config_path}/{module}.json" -e archive_path="data/{config_path}" -v "{data}":/assist/data assist/{module}:{tag}')
print(f"{datetime.now()}")