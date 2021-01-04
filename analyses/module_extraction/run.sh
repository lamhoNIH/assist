#!/bin/bash

if [ ! -d "./Data/module_extraction" ] 
then
    mkdir ./Data/module_extraction
fi

python3 extraction.py --archive_path ${archive_path}
