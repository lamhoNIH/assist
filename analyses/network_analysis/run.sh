#!/bin/bash

python3 analysis_preproc.py --config_file "${config_file}"

Rscript wgcna_codes.R "${config_file}"
