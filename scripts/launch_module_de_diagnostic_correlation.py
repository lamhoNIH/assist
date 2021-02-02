#!/usr/bin/env python3

from datetime import datetime
from os import makedirs, system
from os.path import exists
from shutil import copyfile
from sys import platform

prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

script_dir = os.getcwd()
print(f'{script_dir}')

data = f"{prefix}/Shared drives/NIAAA_ASSIST/Data"
module = "module_de_diagnostic_correlation"

if not exists(f"{data}/{module}"):
    makedirs(f"{data}/{module}")

copyfile(f"{script_dir}/{module}.json", f"{data}/{module}/config.json")

print(f"{datetime.now()}")
system(f'docker run --rm -m 10g -e config_file="{module}/config.json" -e archive_path="{module}" -v "{prefix}/Shared drives/NIAAA_ASSIST/Data":/assist/Data assist/{module}:0.1.0')
print(f"{datetime.now()}")