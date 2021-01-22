#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${SCRIPT_DIR}

DATA="/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data"
MODULE="ml_and_critical_gene_identifier"

if [ ! -d "${DATA}/${MODULE}" ] 
then
    mkdir "${DATA}/${MODULE}"
fi

cp "${SCRIPT_DIR}/${MODULE}".json "${DATA}/${MODULE}"/config.json

date
docker run --rm -m 16g -e config_file="${MODULE}/config.json" -e archive_path="${MODULE}" -v "/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data":/assist/Data assist/${MODULE}:0.1.0
date