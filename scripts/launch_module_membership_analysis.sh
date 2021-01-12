#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${SCRIPT_DIR}

DATA="/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data"
MODULE="module_membership_analysis"

if [ ! -d "${DATA}/${MODULE}" ] 
then
    mkdir "${DATA}/${MODULE}"
fi

cp "${SCRIPT_DIR}/${MODULE}".json "${DATA}/${MODULE}"/config.json

# Takes about 13 minutes to run
date
docker run -m 16g --rm -e config_file="${MODULE}/config.json" -e archive_path="${MODULE}/run2" -v "${DATA}":/assist/Data assist/${MODULE}:0.1.0
date