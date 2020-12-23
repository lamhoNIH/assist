#!/bin/bash

if [ ! -d "./Data/network_analysis" ] 
then
    mkdir ./Data/network_analysis
fi

python3 prepare_network_ids.py

Rscript wgcna_codes.R
