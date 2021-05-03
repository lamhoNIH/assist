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

data = f"{prefix}/Shared drives/NIAAA_ASSIST/Data"
dataset = sys.argv[1]
module = "network_embedding"

config_path = f"pipeline/{dataset}/{module}"

if not exists(f"{data}/{config_path}"):
    makedirs(f"{data}/{config_path}")

copyfile(f"{script_dir}/{dataset}/{module}.json", f"{data}/{config_path}/{module}.json")

print(f"{datetime.now()}")
system(f'docker run --rm -m 28g -e config_file="{data}/{config_path}/{module}.json" -v "{data}":"{data}" assist/{module}:0.1.0')
print(f"{datetime.now()}")