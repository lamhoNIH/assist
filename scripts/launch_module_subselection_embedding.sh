#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${SCRIPT_DIR}

DATA="/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data"
MODULE="module_subselection_embedding"

if [ ! -d "${DATA}/${MODULE}" ] 
then
    mkdir "${DATA}/${MODULE}"
fi

cp "${SCRIPT_DIR}/${MODULE}".json "${DATA}/${MODULE}"/config.json

date
# tried 32G and the process got killed. The process took 8.5 hours on Macbook Pro with 2.4 GHz 8-Core Intel Core i9 and 64 GB 2667 MHz DDR4
docker run --rm -m 40g -e config_file="${MODULE}/config.json" -e archive_path="${MODULE}" -e run_num="run4" -v "/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data":/assist/Data assist/${MODULE}:0.1.0
date