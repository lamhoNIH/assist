#!/usr/bin/env python3

import sys

from datetime import datetime
from os import getcwd, makedirs, system
from os.path import exists
from shutil import copyfile
from sys import platform

if len(sys.argv) != 2:
    print("python launch_ml_and_critical_gene_identifier.py <dataset>")
    exit(2)

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

script_dir = getcwd()
print(f'{script_dir}')

data = f"{prefix}/Shared drives/NIAAA_ASSIST/Data"
dataset = sys.argv[1]
module = "ml_and_critical_gene_identifier"

output_path = f"pipeline/{dataset}/{module}"

if not exists(f"{data}/{output_path}"):
    makedirs(f"{data}/{output_path}")

copyfile(f"{script_dir}/{dataset}/{module}.json", f"{data}/{output_path}/config.json")

print(f"{datetime.now()}")
system(f'docker run --rm -m 16g -e config_file="Data/{output_path}/config.json" -e archive_path="Data/{output_path}" -e run_num="run3" -v "{data}":/assist/Data assist/{module}:0.1.0')
print(f"{datetime.now()}")