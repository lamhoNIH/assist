#!/bin/bash

if [[ ${skip_preproc} != true && ${skip_preproc} != True ]]
then
	python3 preproc.py --config_file ${config_file} --archive_path ${archive_path}
fi

Rscript wgcna_codes.R ${config_file} ${archive_path}
