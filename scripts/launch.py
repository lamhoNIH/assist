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
# human dataset:
# network_analysis: 2021-05-04 16:26:55.893449 to 2021-05-04 16:30:45.484398 -> 3:50, 2nd try 2021-05-05 08:00:10.054772 to 2021-05-05 08:03:28.425164 -> 3:18
# module_extraction: 2021-05-04 18:40:12.848314 to 2021-05-04 18:54:54.975768 -> 14:42
# module_membership_analysis: 2021-05-04 19:10:52.902437 to 2021-05-04 19:24:37.833244 -> 13:32
# module_de_diagnostic_correlation: 2021-05-04 20:33:42.342007 to 2021-05-04 20:50:23.968761 -> 16:41
# network_embedding: 2021-05-05 08:22:05.571182 to 2021-05-05 10:48:19.088962 -> 2:26:14 need to rerun
# ml_and_critical_gene_identifier: 2021-05-04 21:52:04.056219 to 2021-05-04 22:06:18.707775 -> 14:14, need rerun 2021-05-05 11:03:35.929050 to 2021-05-05 11:14:21.570006 -> 10:46
# mouse dataset:
# network_analysis: 2021-05-05 11:06:19.364316 to 2021-05-05 11:28:00.291174 -> 11:41, 2021-05-06 11:56:02.945903 to 2021-05-06 12:15:00.505079 -> 18:58
# module_extraction: 2021-05-05 15:37:01.414458 to 2021-05-05 15:42:34.867059 -> 5:33, need rerun 2021-05-06 12:22:24.000363 to 2021-05-06 12:27:29.887501 -> 05:05
# module_membership_analysis 2021-05-05 15:50:42.149537 to 2021-05-05 15:50:46.879822 -> 3
# module_de_diagnostic_correlation2021-05-05 15:52:18.791553 to 2021-05-05 15:52:26.050638 -> 8
# network_embedding 2021-05-05 15:53:05.549459 to 2021-05-05 16:55:34.308050 -> 1:02:29
# ml_and_critical_gene_identifier 

module_map = {
    "network_analysis": ["10g", "0.1.0"], # ~4 minutes on human
    "module_extraction": ["28g", "0.1.0"],
    "module_membership_analysis": ["16g", "0.1.0"],
    "module_de_diagnostic_correlation": ["10g", "0.1.0"],
    "network_embedding": ["28g", "0.1.0"],
	"ml_and_critical_gene_identifier": ["16g", "0.1.0"]
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

copyfile(f"{script_dir}/{dataset}/{module}.json", f"{data}/{config_path}/{module}.json")

memory = module_map[module][0]
tag = module_map[module][1]
print(f"{datetime.now()}")
system(f'docker run --rm -m {memory} -e config_file="data/{config_path}/{module}.json" -e archive_path="data/{config_path}" -v "{data}":/assist/data assist/{module}:{tag}')
print(f"{datetime.now()}")