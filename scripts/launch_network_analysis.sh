#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo ${SCRIPT_DIR}

DATA="/Volumes/GoogleDrive/Shared drives/NIAAA_ASSIST/Data"
if [[ $1 = "human" || $1 = "mouse" ]]
then
	DATASET=$1
else
	echo "dataset must be either human or mouse"
	exit
fi
MODULE="network_analysis"

OUTPUT_PATH="pipeline/${DATASET}/${MODULE}"

if [ ! -d "${DATA}/${OUTPUT_PATH}" ]
then
    mkdir "${DATA}/${OUTPUT_PATH}"
fi

cp "${SCRIPT_DIR}/${DATASET}/${MODULE}".json "${DATA}/${OUTPUT_PATH}"/config.json

date
docker run --rm -m 10g -e config_file="Data/${OUTPUT_PATH}/config.json" -e archive_path="Data/${OUTPUT_PATH}" -v "${DATA}":/assist/Data assist/${MODULE}:0.1.0
date