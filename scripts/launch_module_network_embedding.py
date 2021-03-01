#!/usr/bin/env python3

import os
import sys

from datetime import datetime
from os import makedirs, system
from os.path import exists
from shutil import copyfile
from sys import platform

if len(sys.argv) != 2:
    print("python launch_module_network_embedding.py <dataset>")
    exit(2)

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

script_dir = os.getcwd()
print(f'{script_dir}')

# data = f"{prefix}/Shared drives/NIAAA_ASSIST/Data"
data = 'C:/Users/bbche/Documents/GitRepos/assist/data'
dataset = sys.argv[1]
module = "module_network_embedding"

output_path = f"pipeline/{dataset}/{module}"

if not exists(f"{data}/{output_path}"):
    makedirs(f"{data}/{output_path}")

copyfile(f"{script_dir}/{dataset}/{module}.json", f"{data}/{output_path}/config.json")

print(f"{datetime.now()}")
system(f'docker run --rm -m 30g -e config_file="Data/{output_path}/config.json" -e archive_path="Data/{output_path}" -e run_num="run1" -v "{data}":/assist/Data assist/{module}:0.1.0')
print(f"{datetime.now()}")