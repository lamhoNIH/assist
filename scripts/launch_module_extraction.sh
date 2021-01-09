#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${SCRIPT_DIR}

DATA="/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data"
MODULE="module_extraction"

if [ ! -d "${DATA}/${MODULE}" ] 
then
    mkdir "${DATA}/${MODULE}"
fi

cp "${SCRIPT_DIR}/${MODULE}".json "${DATA}/${MODULE}"/config.json

# Takes about 8 minutes to run
date
# tried 20G and the process got killed
docker run -m32g --rm -e config_file="${MODULE}/config.json" -e archive_path="${MODULE}/run4" -v "/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data":/assist/Data assist/${MODULE}:0.1.0
date