#!/bin/bash

python3 preproc.py --config_file ${config_file} --archive_path ${archive_path}

Rscript wgcna_codes.R ${config_file} ${archive_path}
